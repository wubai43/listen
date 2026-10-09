"""
Windows 游戏短音效实时识别器

主要功能：
1. 实时监听 Windows 系统播放的音频。
2. 识别 0.1～1 秒左右的游戏音效。
3. 使用 Log-Mel 频谱、频谱变化和能量包络。
4. 在实时音频中滑动搜索短音效，而不是直接比较整个窗口。
5. 通过最佳相似度和第一、第二候选的分差减少误报。
6. 使用后台线程持续采集音频，减少识别计算阻塞录音的问题。

安装：
    pip install numpy scipy librosa soundcard

目录：
    audio_fingerprint_listener.py
    config.json                  # 可选
    VoiceSource/
        data.json
        effect1.wav
        effect2.mp3
"""

from __future__ import annotations

import json
import math
import queue
import sys
import threading
import time
import warnings

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundcard as sc

from scipy.signal import resample_poly


# ============================================================
# 1. 路径与全局配置
# ============================================================

BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)

VOICE_DIR = BASE_DIR / "VoiceSource"
DATA_PATH = VOICE_DIR / "data.json"
CONFIG_PATH = BASE_DIR / "config.json"


DEFAULT_CONFIG = {
    # 特征提取采样率
    "sample_rate": 22050,

    # Windows 系统音频采集采样率
    "capture_sample_rate": 48000,

    # 每次采集时长
    "capture_seconds": 0.05,

    # 每隔多久分析一次
    "analysis_interval": 0.08,

    # 实时匹配缓冲区
    "pre_roll_seconds": 0.15,
    "post_roll_seconds": 1.10,

    # Log-Mel 特征
    "n_fft": 1024,
    "hop_length": 220,
    "n_mels": 48,

    # 最近 120ms 音频的 RMS 门槛
    "energy_threshold": 0.001,

    # 最低相似度
    "min_similarity": 0.70,

    # 第一名和第二名的最低分差
    "min_margin": 0.045,

    # 短音效默认立即确认
    "confirm_count": 1,

    # 识别成功后的冷却时间
    "cooldown_seconds": 0.25,

    # 尝试适应轻微时长变化
    "time_scales": [0.92, 1.0, 1.08],

    # 后台采集队列容量
    "capture_queue_size": 20,

    # 是否输出调试信息
    "debug": True,
}


def log(message: str) -> None:
    print(
        time.strftime("[%H:%M:%S]"),
        message,
        flush=True
    )


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()

    if CONFIG_PATH.exists():
        try:
            user_config = json.loads(
                CONFIG_PATH.read_text(encoding="utf-8")
            )

            if isinstance(user_config, dict):
                config.update(user_config)

        except Exception as exc:
            log(f"[警告] config.json 读取失败：{exc}")

    return config


CFG = load_config()

SR = int(CFG["sample_rate"])
CAPTURE_SR = int(CFG["capture_sample_rate"])

CAPTURE_SECONDS = float(CFG["capture_seconds"])
ANALYSIS_INTERVAL = float(CFG["analysis_interval"])

PRE_ROLL = float(CFG["pre_roll_seconds"])
POST_ROLL = float(CFG["post_roll_seconds"])

N_FFT = int(CFG["n_fft"])
HOP = int(CFG["hop_length"])
N_MELS = int(CFG["n_mels"])

ENERGY_THRESHOLD = float(CFG["energy_threshold"])
MIN_SIMILARITY = float(CFG["min_similarity"])
MIN_MARGIN = float(CFG["min_margin"])

CONFIRM_COUNT = max(
    1,
    int(CFG["confirm_count"])
)

COOLDOWN = float(CFG["cooldown_seconds"])

TIME_SCALES = [
    float(x)
    for x in CFG["time_scales"]
]

QUEUE_SIZE = max(
    2,
    int(CFG["capture_queue_size"])
)

DEBUG = bool(CFG["debug"])


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class Template:
    name: str
    payload: Any

    # 特征矩阵：特征维度 × 时间帧
    feature: np.ndarray

    duration: float


# ============================================================
# 3. 音频处理工具
# ============================================================

def to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(
        samples,
        dtype=np.float32
    )

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return np.nan_to_num(
        samples.reshape(-1),
        copy=False
    )


def resample_audio(
    samples: np.ndarray,
    source_sr: int,
    target_sr: int
) -> np.ndarray:

    samples = to_mono(samples)

    if source_sr == target_sr:
        return samples

    gcd = math.gcd(
        source_sr,
        target_sr
    )

    up = target_sr // gcd
    down = source_sr // gcd

    return resample_poly(
        samples,
        up,
        down
    ).astype(np.float32)


def update_ring(
    ring: np.ndarray,
    block: np.ndarray
) -> None:
    """将新音频写入固定长度滚动缓冲区。"""

    n = len(block)

    if n == 0:
        return

    if n >= len(ring):
        ring[:] = block[-len(ring):]
    else:
        ring[:-n] = ring[n:]
        ring[-n:] = block


def update_ring_with_gap(
    ring: np.ndarray,
    block: np.ndarray,
    missing_samples: int
) -> None:
    """
    如果后台队列丢弃了旧音频块，用零填充大致的时间缺口。
    这样不会把缺口前后的两段声音直接拼接成连续音频。
    """

    missing_samples = max(
        0,
        int(missing_samples)
    )

    if missing_samples > 0:
        zeros = np.zeros(
            min(missing_samples, len(ring)),
            dtype=np.float32
        )
        update_ring(ring, zeros)

    update_ring(ring, block)


# ============================================================
# 4. 短音效 Log-Mel 特征
# ============================================================

def extract_features(
    samples: np.ndarray
) -> np.ndarray:
    """
    返回形状为：
        特征维度 × 时间帧

    特征包括：
        1. Log-Mel 频谱形状
        2. 频谱随时间的变化
        3. 能量包络
        4. 起音变化

    设计重点：
        尽可能保留短音效的起音、音色和衰减特点。
    """

    samples = to_mono(samples)

    if len(samples) < N_FFT:
        return np.empty(
            (N_MELS * 2 + 2, 0),
            dtype=np.float32
        )

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=70,
        fmax=min(
            10000,
            SR // 2
        ),
        power=2.0
    )

    logmel = np.log(
        np.maximum(mel, 1e-10)
    )

    # 每帧的频谱形状
    shape = logmel - np.mean(
        logmel,
        axis=0,
        keepdims=True
    )

    shape /= np.maximum(
        np.linalg.norm(
            shape,
            axis=0,
            keepdims=True
        ),
        1e-8
    )

    # 频谱变化信息
    delta = np.diff(
        logmel,
        axis=1,
        prepend=logmel[:, :1]
    )

    delta -= np.mean(
        delta,
        axis=0,
        keepdims=True
    )

    delta /= np.maximum(
        np.linalg.norm(
            delta,
            axis=0,
            keepdims=True
        ),
        1e-8
    )

    # 能量包络
    rms = librosa.feature.rms(
        y=samples,
        frame_length=N_FFT,
        hop_length=HOP,
        center=True
    )[0]

    energy = np.log(
        np.maximum(rms, 1e-8)
    )

    if len(energy) != shape.shape[1]:
        energy = np.interp(
            np.linspace(
                0,
                1,
                shape.shape[1]
            ),
            np.linspace(
                0,
                1,
                len(energy)
            ),
            energy
        )

    energy -= np.median(energy)

    energy /= max(
        float(np.std(energy)),
        1e-5
    )

    energy = np.clip(
        energy / 3.0,
        -1.0,
        1.0
    )[None, :]

    # 频谱正向变化，用于辅助检测起音
    flux = np.maximum(
        0.0,
        np.diff(
            logmel,
            axis=1,
            prepend=logmel[:, :1]
        )
    ).mean(axis=0)

    flux /= max(
        float(np.max(flux)),
        1e-8
    )

    flux = (
        flux * 2.0 - 1.0
    )[None, :]

    feature = np.concatenate(
        [
            shape,
            0.30 * delta,
            0.25 * energy,
            0.20 * flux
        ],
        axis=0
    )

    feature /= np.maximum(
        np.linalg.norm(
            feature,
            axis=0,
            keepdims=True
        ),
        1e-8
    )

    return feature.astype(np.float32)


# ============================================================
# 5. 加载模板库
# ============================================================

def load_templates(
    data: dict
) -> list[Template]:

    templates = []

    for filename, payload in data.items():

        audio_path = VOICE_DIR / filename

        if not audio_path.exists():

            alternative = audio_path.with_suffix(
                ".mp3"
                if audio_path.suffix.lower() == ".mp4"
                else ".mp4"
            )

            if alternative.exists():
                audio_path = alternative
            else:
                log(
                    f"[警告] 找不到模板：{audio_path}"
                )
                continue

        try:

            audio, _ = librosa.load(
                str(audio_path),
                sr=SR,
                mono=True
            )

            duration = len(audio) / SR
            feature = extract_features(audio)

            if feature.shape[1] < 4:
                log(
                    f"[警告] 模板过短，无法提取有效特征：{filename}"
                )
                continue

            templates.append(
                Template(
                    name=filename,
                    payload=payload,
                    feature=feature,
                    duration=duration
                )
            )

            log(
                f"[模板] {filename} | "
                f"时长={duration:.3f}s | "
                f"帧数={feature.shape[1]}"
            )

        except Exception as exc:

            log(
                f"[错误] 模板加载失败 {filename}："
                f"{type(exc).__name__}: {exc}"
            )

    return templates


# ============================================================
# 6. 滑动窗口匹配
# ============================================================

def scale_template(
    feature: np.ndarray,
    scale: float
) -> np.ndarray:

    if abs(scale - 1.0) < 1e-6:
        return feature

    old_length = feature.shape[1]

    new_length = max(
        3,
        int(round(old_length * scale))
    )

    old_x = np.linspace(
        0.0,
        1.0,
        old_length
    )

    new_x = np.linspace(
        0.0,
        1.0,
        new_length
    )

    result = np.stack(
        [
            np.interp(
                new_x,
                old_x,
                row
            )
            for row in feature
        ],
        axis=0
    ).astype(np.float32)

    result /= np.maximum(
        np.linalg.norm(
            result,
            axis=0,
            keepdims=True
        ),
        1e-8
    )

    return result


def sliding_similarity(
    live: np.ndarray,
    template: np.ndarray
) -> float:
    """
    把整个模板放到实时窗口的所有可能位置进行比较。
    不要求实时窗口和模板具有相同长度。
    """

    if (
        live.shape[0] != template.shape[0]
        or live.shape[1] < 3
        or template.shape[1] < 3
    ):
        return -1.0

    live_length = live.shape[1]
    best_score = -1.0

    for scale in TIME_SCALES:

        target = scale_template(
            template,
            scale
        )

        target_length = target.shape[1]

        if target_length > live_length:
            continue

        # 形状为：
        # 特征维度 × 可能的开始位置 × 模板帧数
        windows = np.lib.stride_tricks.sliding_window_view(
            live,
            window_shape=target_length,
            axis=1
        )

        scores = np.einsum(
            "dkl,dl->k",
            windows,
            target,
            optimize=True
        )

        scores /= target_length

        current_score = float(
            np.max(scores)
        )

        best_score = max(
            best_score,
            current_score
        )

    return float(
        np.clip(
            best_score,
            -1.0,
            1.0
        )
    )


# ============================================================
# 7. 后台音频采集线程
# ============================================================

class AudioCaptureThread(threading.Thread):

    def __init__(
        self,
        device,
        stop_event: threading.Event,
        audio_queue: queue.Queue
    ):

        super().__init__(daemon=True)

        self.device = device
        self.stop_event = stop_event
        self.audio_queue = audio_queue

        self.error = None

        self.frames = max(
            1,
            int(
                CAPTURE_SR * CAPTURE_SECONDS
            )
        )

    def run(self):

        try:

            with self.device.recorder(
                samplerate=CAPTURE_SR,
                channels=1
            ) as recorder:

                while not self.stop_event.is_set():

                    raw = recorder.record(
                        numframes=self.frames
                    )

                    chunk = resample_audio(
                        raw,
                        CAPTURE_SR,
                        SR
                    )

                    captured_at = time.monotonic()

                    item = (
                        captured_at,
                        chunk
                    )

                    # 队列满时丢弃最旧的块以控制延迟。
                    # 时间戳会用于在分析缓冲区中补出缺失区间。
                    try:

                        self.audio_queue.put_nowait(item)

                    except queue.Full:

                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass

                        try:
                            self.audio_queue.put_nowait(item)
                        except queue.Full:
                            pass

        except Exception as exc:

            self.error = exc
            self.stop_event.set()


def choose_loopback_device():

    try:

        speaker = sc.default_speaker()

        devices = sc.all_microphones(
            include_loopback=True
        )

        for device in devices:

            if (
                getattr(device, "isloopback", False)
                and speaker.name.lower()
                in device.name.lower()
            ):
                return device

        for device in devices:

            if getattr(
                device,
                "isloopback",
                False
            ):
                return device

    except Exception as exc:

        log(
            f"[错误] 查询回环设备失败：{exc}"
        )

    return None


# ============================================================
# 8. 实时识别主循环
# ============================================================

def run_listener(
    templates: list[Template]
) -> None:

    device = choose_loopback_device()

    if device is None:
        raise RuntimeError(
            "找不到 Windows 系统回环录音设备。"
            "请检查默认播放设备及 soundcard 安装。"
        )

    log(
        f"[监听] 设备：{device.name}"
    )

    longest_template = max(
        template.duration
        for template in templates
    )

    # 预留最长模板、前置音频以及少量余量
    ring_seconds = max(
        PRE_ROLL + POST_ROLL,
        PRE_ROLL + longest_template * 1.15 + 0.15
    )

    ring = np.zeros(
        int(SR * ring_seconds),
        dtype=np.float32
    )

    audio_queue = queue.Queue(
        maxsize=QUEUE_SIZE
    )

    stop_event = threading.Event()

    capture_thread = AudioCaptureThread(
        device=device,
        stop_event=stop_event,
        audio_queue=audio_queue
    )

    capture_thread.start()

    last_analysis = 0.0
    last_output = 0.0
    last_debug = 0.0

    previous_name = None
    consecutive = 0

    last_captured_at = None

    try:

        while not stop_event.is_set():

            try:

                captured_at, block = audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                if capture_thread.error is not None:
                    raise RuntimeError(
                        f"音频采集失败：{capture_thread.error}"
                    )

                continue

            # 检查队列是否因分析压力丢失了音频时间段。
            missing_samples = 0

            if last_captured_at is not None:

                elapsed = (
                    captured_at - last_captured_at
                )

                expected_samples = int(
                    elapsed * SR
                )

                missing_samples = max(
                    0,
                    expected_samples - len(block)
                )

                # 微小调度抖动不需要插入缺口
                if missing_samples < int(SR * 0.008):
                    missing_samples = 0

            last_captured_at = captured_at

            update_ring_with_gap(
                ring,
                block,
                missing_samples
            )

            now = time.monotonic()

            if now - last_analysis < ANALYSIS_INTERVAL:
                continue

            last_analysis = now

            if now - last_output < COOLDOWN:
                continue

            # 检测最近一小段音频，而不是整个缓冲区的平均音量。
            recent_count = min(
                len(ring),
                int(SR * 0.12)
            )

            recent = ring[-recent_count:]

            rms = float(
                np.sqrt(
                    np.mean(recent * recent)
                )
            )

            if rms < ENERGY_THRESHOLD:

                previous_name = None
                consecutive = 0
                continue

            live_feature = extract_features(
                ring
            )

            candidates = []

            for template in templates:

                score = sliding_similarity(
                    live_feature,
                    template.feature
                )

                candidates.append(
                    (score, template)
                )

            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            if not candidates:
                continue

            best_score, best_template = candidates[0]

            second_score = (
                candidates[1][0]
                if len(candidates) > 1
                else -1.0
            )

            margin = best_score - second_score

            if DEBUG and now - last_debug >= 0.5:

                candidate_text = " | ".join(
                    f"{template.name}: {score:.3f}"
                    for score, template
                    in candidates[:5]
                )

                log(
                    f"[候选] RMS={rms:.4f} | "
                    f"{candidate_text} | "
                    f"分差={margin:.3f}"
                )

                last_debug = now

            # 绝对分数和候选分差必须同时满足
            accepted = (
                best_score >= MIN_SIMILARITY
                and margin >= MIN_MARGIN
            )

            if not accepted:

                previous_name = None
                consecutive = 0
                continue

            if previous_name == best_template.name:

                consecutive += 1

            else:

                previous_name = best_template.name
                consecutive = 1

            if consecutive < CONFIRM_COUNT:
                continue

            log("=" * 60)

            log(
                f"[识别成功] {best_template.name} | "
                f"similarity={best_score:.3f} | "
                f"margin={margin:.3f} | "
                f"RMS={rms:.4f}"
            )

            log(
                "[关联数据] "
                + json.dumps(
                    best_template.payload,
                    ensure_ascii=False
                )
            )

            log("=" * 60)

            last_output = now
            previous_name = None
            consecutive = 0

            # 不清空整个 ring，避免丢掉紧接着出现的短音效。
            # 冷却时间负责抑制同一音效重复触发。

    finally:

        stop_event.set()
        capture_thread.join(timeout=2.0)

    if capture_thread.error is not None:
        raise RuntimeError(
            f"后台采集线程异常：{capture_thread.error}"
        )


# ============================================================
# 9. 主程序
# ============================================================

def main() -> int:

    if not DATA_PATH.exists():

        log(
            f"[错误] 找不到：{DATA_PATH}"
        )

        return 2

    try:

        data = json.loads(
            DATA_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        log(
            f"[错误] data.json 读取失败：{exc}"
        )

        return 2

    if not isinstance(data, dict) or not data:

        log(
            "[错误] data.json 必须是非空对象，"
            "键为音频文件名，值为识别成功后输出的数据。"
        )

        return 2

    templates = load_templates(data)

    if not templates:

        log(
            "[错误] 没有加载成功的模板音频。"
        )

        return 2

    log(
        f"[就绪] 已加载 {len(templates)} 个模板。"
    )

    log(
        f"[参数] SR={SR} | "
        f"analysis_interval={ANALYSIS_INTERVAL} | "
        f"min_similarity={MIN_SIMILARITY} | "
        f"min_margin={MIN_MARGIN} | "
        f"cooldown={COOLDOWN}"
    )

    try:

        run_listener(templates)

    except KeyboardInterrupt:

        log("用户停止监听。")

    except Exception as exc:

        log(
            f"[错误] {type(exc).__name__}: {exc}"
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())