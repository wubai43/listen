# -*- coding: utf-8 -*-
"""
游戏短音效识别工具
Python + PySide6 + librosa + soundcard

功能：
1. 实时监听 Windows 系统播放音频。
2. 匹配 0.1～1 秒的游戏短音效。
3. 使用 Log-Mel 频谱、频谱变化、能量包络进行滑动匹配。
4. 使用最佳相似度和第一、第二候选分差控制误报。
5. 保留原有四个 UI 页面：
   - 识别运行
   - 识别参数
   - 物品数据
   - 音频文件
6. 支持导入、删除、试听模板音频。
7. 支持编辑 data.json 和 config.json。
8. 使用独立音频采集线程，降低采集被计算阻塞的风险。

依赖安装：
    pip install PySide6 numpy scipy librosa soundcard

目录结构：
    main.py
    config.json                   # 程序自动生成
    VoiceSource/
        data.json
        sound1.wav
        sound2.mp3
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import sys
import time
import threading
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import librosa
import soundcard as sc

from scipy.signal import resample_poly

from PySide6.QtCore import (
    Qt,
    Signal,
    QThread,
    QUrl,
)

from PySide6.QtGui import QFont

from PySide6.QtMultimedia import (
    QMediaPlayer,
    QAudioOutput,
)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QTabWidget,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QCheckBox,
)


# ============================================================
# 1. 路径和默认参数
# ============================================================

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

VOICE_DIR = BASE_DIR / "VoiceSource"
JSON_PATH = VOICE_DIR / "data.json"
CONFIG_PATH = BASE_DIR / "config.json"

AUDIO_EXTS = (
    ".mp3",
    ".mp4",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
)

DEFAULT_CONFIG = {
    # 特征分析采样率
    "sample_rate": 22050,

    # Windows 系统回环录音采样率
    "capture_sample_rate": 48000,

    # 每次录音时长
    "capture_seconds": 0.05,

    # 两次特征比较的时间间隔
    "analysis_interval": 0.08,

    # 实时缓冲区保留时间
    "pre_roll_seconds": 0.15,
    "post_roll_seconds": 1.10,

    # 特征参数
    "n_fft": 1024,
    "hop_length": 220,
    "n_mels": 48,

    # 最近音频的 RMS 音量门槛
    "energy_threshold": 0.001,

    # 最低匹配相似度
    "min_similarity": 0.70,

    # 第一名与第二名之间的最低分差
    "min_margin": 0.045,

    # 默认立即确认，适合较短的游戏音效
    "confirm_count": 1,

    # 同一音效重复触发抑制时间
    "cooldown_seconds": 0.10,

    # 时长变化适配
    "time_scales": [0.92, 1.0, 1.08],

    # 采集队列容量
    "capture_queue_size": 20,

    # 是否显示调试信息
    "debug": True,
}


# ============================================================
# 2. 通用工具
# ============================================================

def log_time() -> str:
    return time.strftime("[%H:%M:%S]")


def app_log(message: str) -> None:
    print(log_time(), message, flush=True)


def load_json(path: Path, default: Any) -> Any:
    """读取 JSON，文件不存在时返回默认值。"""

    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except Exception as exc:
        QMessageBox.warning(
            None,
            "读取失败",
            f"读取文件失败：\n{path}\n\n{exc}",
        )

        return default


def save_json(path: Path, data: Any) -> None:
    """安全写入 JSON，避免中途写入导致原文件损坏。"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    os.replace(
        str(temp_path),
        str(path),
    )


def list_audio_files(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []

    return sorted(
        filename
        for filename in os.listdir(directory)
        if Path(filename).suffix.lower() in AUDIO_EXTS
    )


def resolve_audio_path(filename: str) -> Path | None:
    """支持文件扩展名不同、但文件基名相同的情形。"""

    path = VOICE_DIR / filename

    if path.exists():
        return path

    base = Path(filename).stem.lower()

    for actual in list_audio_files(VOICE_DIR):
        if Path(actual).stem.lower() == base:
            return VOICE_DIR / actual

    return None


def find_associated_key(
    filename: str,
    data: dict,
) -> str | None:

    if filename in data:
        return filename

    base = Path(filename).stem.lower()

    for key in data:
        if Path(key).stem.lower() == base:
            return key

    return None


def is_audio_associated(
    filename: str,
    data: dict,
) -> bool:

    return find_associated_key(
        filename,
        data,
    ) is not None


def to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return np.nan_to_num(
        samples.reshape(-1),
        copy=False,
    )


def resample_audio(
    samples: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    对完整音频块进行重采样。
    实时识别中先在录音采样率下拼接，再对整个缓冲区重采样，
    避免每个小录音块边缘出现重复滤波影响。
    """

    samples = to_mono(samples)

    if source_sr == target_sr:
        return samples

    divisor = math.gcd(
        source_sr,
        target_sr,
    )

    return resample_poly(
        samples,
        target_sr // divisor,
        source_sr // divisor,
    ).astype(np.float32)


def update_ring(
    ring: np.ndarray,
    block: np.ndarray,
) -> None:
    """更新固定长度滚动缓冲区。"""

    n = len(block)

    if n <= 0:
        return

    if n >= len(ring):
        ring[:] = block[-len(ring):]

    else:
        ring[:-n] = ring[n:]
        ring[-n:] = block


def update_ring_with_gap(
    ring: np.ndarray,
    block: np.ndarray,
    gap_samples: int,
) -> None:
    """
    对发现的录音块缺口填零，避免将缺口两侧音频直接拼接。
    """

    gap_samples = max(
        0,
        int(gap_samples),
    )

    if gap_samples:
        zeros = np.zeros(
            min(gap_samples, len(ring)),
            dtype=np.float32,
        )

        update_ring(
            ring,
            zeros,
        )

    update_ring(
        ring,
        block,
    )


# ============================================================
# 3. 音频匹配引擎
# ============================================================

@dataclass
class Template:
    name: str
    payload: Any
    feature: np.ndarray
    duration: float


class AudioMatcher:
    """
    短游戏音效模板匹配器。

    使用逐帧归一化的 Log-Mel 特征，
    在实时音频窗口中滑动搜索模板。
    """

    def __init__(
        self,
        config: dict,
        log_callback,
    ):

        self.cfg = config
        self.emit_log = log_callback

        self.sr = int(
            config["sample_rate"]
        )

        self.capture_sr = int(
            config["capture_sample_rate"]
        )

        self.n_fft = int(
            config["n_fft"]
        )

        self.hop = int(
            config["hop_length"]
        )

        self.n_mels = int(
            config["n_mels"]
        )

        self.time_scales = [
            float(value)
            for value in config["time_scales"]
        ]

    def log(self, message: str) -> None:
        self.emit_log(
            f"{log_time()} {message}"
        )

    # --------------------------------------------------------
    # 特征提取
    # --------------------------------------------------------

    def extract_features(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        audio = to_mono(audio)

        if len(audio) < self.n_fft:
            return np.empty(
                (self.n_mels * 2 + 2, 0),
                dtype=np.float32,
            )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmin=70,
            fmax=min(
                10000,
                self.sr // 2,
            ),
            power=2.0,
        )

        # 对数能量频谱
        logmel = np.log(
            np.maximum(
                mel,
                1e-10,
            )
        )

        # 频谱形状
        shape = logmel - np.mean(
            logmel,
            axis=0,
            keepdims=True,
        )

        shape /= np.maximum(
            np.linalg.norm(
                shape,
                axis=0,
                keepdims=True,
            ),
            1e-8,
        )

        # 相邻帧频谱变化
        delta = np.diff(
            logmel,
            axis=1,
            prepend=logmel[:, :1],
        )

        delta -= np.mean(
            delta,
            axis=0,
            keepdims=True,
        )

        delta /= np.maximum(
            np.linalg.norm(
                delta,
                axis=0,
                keepdims=True,
            ),
            1e-8,
        )

        # 能量包络
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop,
            center=True,
        )[0]

        energy = np.log(
            np.maximum(
                rms,
                1e-8,
            )
        )

        if len(energy) != shape.shape[1]:
            energy = np.interp(
                np.linspace(
                    0,
                    1,
                    shape.shape[1],
                ),
                np.linspace(
                    0,
                    1,
                    len(energy),
                ),
                energy,
            )

        energy -= np.median(energy)

        energy /= max(
            float(np.std(energy)),
            1e-5,
        )

        energy = np.clip(
            energy / 3.0,
            -1.0,
            1.0,
        )[None, :]

        # 频谱起音变化
        flux = np.maximum(
            0.0,
            np.diff(
                logmel,
                axis=1,
                prepend=logmel[:, :1],
            ),
        ).mean(axis=0)

        flux /= max(
            float(np.max(flux)),
            1e-8,
        )

        flux = (
            flux * 2.0 - 1.0
        )[None, :]

        # 组合特征
        feature = np.concatenate(
            [
                shape,
                0.30 * delta,
                0.25 * energy,
                0.20 * flux,
            ],
            axis=0,
        )

        # 对各帧统一归一化
        feature /= np.maximum(
            np.linalg.norm(
                feature,
                axis=0,
                keepdims=True,
            ),
            1e-8,
        )

        return feature.astype(
            np.float32
        )

    # --------------------------------------------------------
    # 加载模板
    # --------------------------------------------------------

    def load_templates(
        self,
        data: dict,
    ) -> list[Template]:

        templates = []

        for filename, payload in data.items():

            audio_path = resolve_audio_path(
                filename
            )

            if audio_path is None:
                self.log(
                    f"[警告] 找不到模板文件：{filename}"
                )
                continue

            try:

                audio, _ = librosa.load(
                    str(audio_path),
                    sr=self.sr,
                    mono=True,
                )

                duration = len(audio) / self.sr

                feature = self.extract_features(
                    audio
                )

                if feature.shape[1] < 3:
                    self.log(
                        f"[警告] 模板没有足够特征：{filename}"
                    )
                    continue

                templates.append(
                    Template(
                        name=filename,
                        payload=payload,
                        feature=feature,
                        duration=duration,
                    )
                )

                self.log(
                    f"[模板] {filename} | "
                    f"{duration:.3f}s | "
                    f"{feature.shape[1]} 帧"
                )

            except Exception as exc:

                self.log(
                    f"[错误] 加载 {filename} 失败："
                    f"{type(exc).__name__}: {exc}"
                )

        return templates

    # --------------------------------------------------------
    # 时间缩放
    # --------------------------------------------------------

    @staticmethod
    def scale_template(
        feature: np.ndarray,
        scale: float,
    ) -> np.ndarray:

        if abs(scale - 1.0) < 1e-6:
            return feature

        old_length = feature.shape[1]

        new_length = max(
            3,
            int(round(old_length * scale)),
        )

        old_x = np.linspace(
            0.0,
            1.0,
            old_length,
        )

        new_x = np.linspace(
            0.0,
            1.0,
            new_length,
        )

        scaled = np.stack(
            [
                np.interp(
                    new_x,
                    old_x,
                    row,
                )
                for row in feature
            ],
            axis=0,
        ).astype(np.float32)

        scaled /= np.maximum(
            np.linalg.norm(
                scaled,
                axis=0,
                keepdims=True,
            ),
            1e-8,
        )

        return scaled

    # --------------------------------------------------------
    # 滑动窗口匹配
    # --------------------------------------------------------

    def sliding_similarity(
        self,
        live: np.ndarray,
        template: np.ndarray,
    ) -> tuple[float, int]:

        """
        在实时特征窗口中滑动搜索模板。

        返回：
            最佳相似度
            最佳位置对应的特征帧索引
        """

        if (
            live.shape[0] != template.shape[0]
            or live.shape[1] < 3
            or template.shape[1] < 3
        ):
            return -1.0, 0

        live_length = live.shape[1]

        best_score = -1.0
        best_offset = 0

        for scale in self.time_scales:

            target = self.scale_template(
                template,
                scale,
            )

            target_length = target.shape[1]

            if target_length > live_length:
                continue

            windows = (
                np.lib.stride_tricks
                .sliding_window_view(
                    live,
                    window_shape=target_length,
                    axis=1,
                )
            )

            scores = np.einsum(
                "dkl,dl->k",
                windows,
                target,
                optimize=True,
            )

            scores /= target_length

            max_score = float(
                np.max(scores)
            )

            # 若多个位置分数非常接近，优先选择最近出现的位置。
            # 这有助于避免重复报告缓冲区里较早出现的同一音效。
            possible = np.flatnonzero(
                scores >= max_score - 0.005
            )

            offset = (
                int(possible[-1])
                if len(possible)
                else int(np.argmax(scores))
            )

            if max_score > best_score:

                best_score = max_score
                best_offset = offset

            elif abs(max_score - best_score) <= 0.005:

                if offset > best_offset:
                    best_offset = offset

        return (
            float(
                np.clip(
                    best_score,
                    -1.0,
                    1.0,
                )
            ),
            best_offset,
        )

    # --------------------------------------------------------
    # 搜索回环设备
    # --------------------------------------------------------

    def find_loopback_device(self):

        try:

            speaker = sc.default_speaker()

            devices = sc.all_microphones(
                include_loopback=True
            )

            for device in devices:

                if (
                    getattr(
                        device,
                        "isloopback",
                        False,
                    )
                    and speaker.name.lower()
                    in device.name.lower()
                ):
                    return device

            for device in devices:

                if getattr(
                    device,
                    "isloopback",
                    False,
                ):
                    return device

        except Exception as exc:

            self.log(
                f"[错误] 查询系统音频设备失败：{exc}"
            )

        return None

    # --------------------------------------------------------
    # 实时识别
    # --------------------------------------------------------

    def run_listener(
        self,
        templates: list[Template],
        stop_event: threading.Event,
    ) -> None:

        device = self.find_loopback_device()

        if device is None:
            raise RuntimeError(
                "没有找到 Windows 系统回环录音设备。"
                "请检查系统播放设备和 soundcard 安装。"
            )

        self.log(
            f"[监听] 当前设备：{device.name}"
        )

        max_template_duration = max(
            template.duration
            for template in templates
        )

        # 从当前识别配置中读取缓冲区参数
        pre_roll = float(
            self.cfg.get("pre_roll_seconds", 0.15)
        )

        post_roll = float(
            self.cfg.get("post_roll_seconds", 1.10)
        )

        # 根据最长模板音频长度确定缓冲区大小
        ring_seconds = (
                pre_roll
                + max(
            post_roll,
            max_template_duration * 1.15 + 0.15,
            )
        )
        # 注意：缓冲区使用录音设备采样率，
        # 不在每个极短录音块上单独重采样。
        ring = np.zeros(
            int(
                self.capture_sr * ring_seconds
            ),
            dtype=np.float32,
        )

        audio_queue = queue.Queue(
            maxsize=max(
                2,
                int(
                    self.cfg["capture_queue_size"]
                ),
            )
        )

        capture_thread = AudioCaptureThread(
            device=device,
            capture_sr=self.capture_sr,
            capture_seconds=float(
                self.cfg["capture_seconds"]
            ),
            audio_queue=audio_queue,
            stop_event=stop_event,
            log_callback=self.log,
        )

        capture_thread.start()

        last_analysis = 0.0
        last_debug = 0.0

        last_captured_at = None

        previous_name = None
        consecutive = 0

        # 分别记录每种音效上次触发的事件时间
        last_event_times: dict[str, float] = {}

        frame_seconds = self.hop / self.sr

        min_similarity = float(
            self.cfg["min_similarity"]
        )

        min_margin = float(
            self.cfg["min_margin"]
        )

        energy_threshold = float(
            self.cfg["energy_threshold"]
        )

        analysis_interval = float(
            self.cfg["analysis_interval"]
        )

        cooldown = float(
            self.cfg["cooldown_seconds"]
        )

        confirm_count = max(
            1,
            int(
                self.cfg["confirm_count"]
            ),
        )

        debug = bool(
            self.cfg["debug"]
        )

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

                # 检测队列中的音频时间间隙
                gap_samples = 0

                if last_captured_at is not None:

                    elapsed = (
                        captured_at
                        - last_captured_at
                    )

                    expected = int(
                        elapsed * self.capture_sr
                    )

                    gap_samples = max(
                        0,
                        expected - len(block),
                    )

                    # 忽略非常小的时间调度抖动
                    if gap_samples < int(
                        self.capture_sr * 0.008
                    ):
                        gap_samples = 0

                last_captured_at = captured_at

                update_ring_with_gap(
                    ring,
                    block,
                    gap_samples,
                )

                now = time.monotonic()

                if (
                    now - last_analysis
                    < analysis_interval
                ):
                    continue

                last_analysis = now

                # 仅对当前尾部的声音进行音量门控
                recent_count = min(
                    len(ring),
                    int(
                        self.capture_sr * 0.12
                    ),
                )

                recent = ring[-recent_count:]

                rms = float(
                    np.sqrt(
                        np.mean(recent * recent)
                    )
                )

                if rms < energy_threshold:

                    previous_name = None
                    consecutive = 0

                    continue

                # 对完整滚动缓冲区做重采样，避免小块边界处理造成影响
                live_audio = resample_audio(
                    ring,
                    self.capture_sr,
                    self.sr,
                )

                live_feature = self.extract_features(
                    live_audio
                )

                candidates = []

                for template in templates:

                    score, offset = self.sliding_similarity(
                        live_feature,
                        template.feature,
                    )

                    candidates.append(
                        (
                            score,
                            offset,
                            template,
                        )
                    )

                candidates.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                if not candidates:
                    continue

                best_score, best_offset, best_template = (
                    candidates[0]
                )

                second_score = (
                    candidates[1][0]
                    if len(candidates) > 1
                    else -1.0
                )

                margin = (
                    best_score
                    - second_score
                )

                if (
                    debug
                    and now - last_debug >= 0.5
                ):

                    preview = " | ".join(
                        f"{template.name}: {score:.3f}"
                        for score, _, template
                        in candidates[:5]
                    )

                    self.log(
                        f"[候选] RMS={rms:.4f} | "
                        f"{preview} | "
                        f"分差={margin:.3f}"
                    )

                    last_debug = now

                accepted = (
                    best_score >= min_similarity
                    and margin >= min_margin
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

                if consecutive < confirm_count:
                    continue

                # 估算最佳匹配片段在实时窗口中的起始时间
                live_end_time = captured_at

                event_time = (
                    live_end_time
                    - (
                        live_feature.shape[1]
                        - best_offset
                    ) * frame_seconds
                )

                last_event = last_event_times.get(
                    best_template.name,
                    float("-inf"),
                )

                # 同一段音效仍留在滚动缓冲区时不重复报告
                if event_time > last_event + cooldown:

                    last_event_times[
                        best_template.name
                    ] = event_time

                    self.log("=" * 58)

                    self.log(
                        f"[识别成功] {best_template.name} | "
                        f"相似度={best_score:.3f} | "
                        f"分差={margin:.3f} | "
                        f"RMS={rms:.4f}"
                    )

                    self.log(
                        "[关联数据] "
                        + json.dumps(
                            best_template.payload,
                            ensure_ascii=False,
                        )
                    )

                    self.log("=" * 58)

                previous_name = None
                consecutive = 0

        finally:

            stop_event.set()

            capture_thread.join(
                timeout=2.0
            )

        if capture_thread.error is not None:
            raise RuntimeError(
                f"音频采集线程异常：{capture_thread.error}"
            )


# ============================================================
# 4. 独立音频采集线程
# ============================================================

class AudioCaptureThread(threading.Thread):

    def __init__(
        self,
        device,
        capture_sr: int,
        capture_seconds: float,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        log_callback,
    ):

        super().__init__(
            daemon=True
        )

        self.device = device
        self.capture_sr = capture_sr
        self.capture_seconds = capture_seconds

        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.log_callback = log_callback

        self.error = None

        self.frames = max(
            1,
            int(
                capture_sr
                * capture_seconds
            ),
        )

    def run(self):

        try:

            with self.device.recorder(
                samplerate=self.capture_sr,
                channels=1,
            ) as recorder:

                while not self.stop_event.is_set():

                    block = recorder.record(
                        numframes=self.frames
                    )

                    block = to_mono(block)

                    captured_at = time.monotonic()

                    item = (
                        captured_at,
                        block,
                    )

                    try:

                        self.audio_queue.put_nowait(
                            item
                        )

                    except queue.Full:

                        # 不让旧音频不断堆积导致识别延迟越来越高
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass

                        try:
                            self.audio_queue.put_nowait(
                                item
                            )
                        except queue.Full:
                            pass

        except Exception as exc:

            self.error = exc

            self.stop_event.set()


# ============================================================
# 5. UI 日志重定向
# ============================================================

class RecognitionWorker(QThread):

    log = Signal(str)

    def __init__(
        self,
        config: dict,
        item_data: dict,
    ):

        super().__init__()

        self.config = dict(config)
        self.item_data = dict(item_data)

        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def write_log(self, message: str):
        self.log.emit(message)

    def run(self):

        try:

            self.write_log(
                "[启动] 正在读取模板音频……"
            )

            matcher = AudioMatcher(
                self.config,
                self.write_log,
            )

            templates = matcher.load_templates(
                self.item_data
            )

            if not templates:

                self.write_log(
                    "[错误] 没有有效的模板音频，请检查文件和 data.json。"
                )

                return

            if self.stop_event.is_set():
                return

            self.write_log(
                f"[就绪] 已加载 {len(templates)} 个模板。"
            )

            self.write_log(
                "[启动] 开始实时监听系统音频。"
            )

            matcher.run_listener(
                templates,
                self.stop_event,
            )

        except Exception as exc:

            self.write_log(
                f"[错误] {type(exc).__name__}: {exc}"
            )

        finally:

            self.write_log(
                "[状态] 识别线程已退出。"
            )


# ============================================================
# 6. 主窗口
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "游戏音效识别 · 配置管理"
        )

        self.resize(
            1120,
            780,
        )

        VOICE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.item_data = load_json(
            JSON_PATH,
            {},
        )

        if not isinstance(self.item_data, dict):
            self.item_data = {}

        self.config_data = load_json(
            CONFIG_PATH,
            DEFAULT_CONFIG.copy(),
        )

        if not isinstance(self.config_data, dict):
            self.config_data = DEFAULT_CONFIG.copy()

        self.config_data = {
            **DEFAULT_CONFIG,
            **self.config_data,
        }

        self.current_audio = None
        self.dirty = False

        self.worker = None

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)

        self.player.setAudioOutput(
            self.audio_output
        )

        self.player.mediaStatusChanged.connect(
            self._on_media_status_changed
        )

        self.player.errorOccurred.connect(
            self._on_player_error
        )

        self.player.mediaStatusChanged.connect(
            self._on_media_status_changed
        )

        self._playing_button = None
        self._playing_file = None

        self._build_ui()

        self._load_config_to_widgets()

        self.refresh_audio_list()

        self.refresh_audio_files()

        self.refresh_play_list()

        self._update_status()

    # --------------------------------------------------------
    # 界面总结构
    # --------------------------------------------------------

    def _build_ui(self):

        self.tabs = QTabWidget()

        self.setCentralWidget(
            self.tabs
        )

        self.tabs.addTab(
            self._build_run_tab(),
            "识别运行",
        )

        self.tabs.addTab(
            self._build_config_tab(),
            "识别参数",
        )

        self.tabs.addTab(
            self._build_items_tab(),
            "物品数据",
        )

        self.tabs.addTab(
            self._build_audio_tab(),
            "音频文件",
        )

        self.tabs.currentChanged.connect(
            self._on_tab_changed
        )

        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f6f8;
            }

            QGroupBox {
                border: 1px solid #d7dbe2;
                border-radius: 6px;
                margin-top: 10px;
                padding: 12px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }

            QPushButton {
                padding: 6px 12px;
                border: 1px solid #cbd0d8;
                border-radius: 4px;
                background: #ffffff;
            }

            QPushButton:hover {
                background: #edf3ff;
            }

            QPlainTextEdit, QListWidget, QTableWidget {
                background: #ffffff;
                border: 1px solid #d7dbe2;
                border-radius: 4px;
            }

            QTabWidget::pane {
                border: 1px solid #d7dbe2;
                background: #f5f6f8;
            }

            QTabBar::tab {
                padding: 8px 18px;
            }
            """
        )

    # --------------------------------------------------------
    # 识别运行页
    # --------------------------------------------------------

    def _build_run_tab(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        top = QHBoxLayout()

        self.run_button = QPushButton(
            "开始识别"
        )

        self.run_button.setMinimumHeight(
            38
        )

        self.run_button.clicked.connect(
            self.on_run_clicked
        )

        self.clear_log_btn = QPushButton(
            "清空日志"
        )

        self.clear_log_btn.clicked.connect(
            self.clear_log
        )

        top.addWidget(
            self.run_button
        )

        top.addWidget(
            self.clear_log_btn
        )

        self.run_state_label = QLabel(
            "当前状态：未启动"
        )

        top.addWidget(
            self.run_state_label
        )

        top.addStretch(1)

        layout.addLayout(top)

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        # 左侧模板列表
        left = QWidget()

        left_layout = QVBoxLayout(left)

        left_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        left_layout.addWidget(
            QLabel(
                "音频列表（点击 ▶ 试听）："
            )
        )

        self.play_list = QListWidget()

        self.play_list.setMinimumWidth(
            300
        )

        left_layout.addWidget(
            self.play_list
        )

        # 右侧识别日志
        right = QWidget()

        right_layout = QVBoxLayout(right)

        right_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        right_layout.addWidget(
            QLabel(
                "识别日志与候选排名："
            )
        )

        self.log_view = QPlainTextEdit()

        self.log_view.setReadOnly(
            True
        )

        self.log_view.setFont(
            QFont(
                "Consolas",
                9,
            )
        )

        right_layout.addWidget(
            self.log_view
        )

        splitter.addWidget(left)
        splitter.addWidget(right)

        splitter.setStretchFactor(
            0,
            1,
        )

        splitter.setStretchFactor(
            1,
            2,
        )

        layout.addWidget(
            splitter
        )

        return page

    # --------------------------------------------------------
    # 识别参数页
    # --------------------------------------------------------

    def _build_config_tab(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        group = QGroupBox(
            "短音效匹配参数"
        )

        form = QFormLayout(group)

        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight
        )

        # 特征采样率
        self.sample_rate = QComboBox()

        self.sample_rate.addItems(
            [
                "16000",
                "22050",
                "44100",
            ]
        )

        form.addRow(
            "特征采样率：",
            self.sample_rate,
        )

        # 录音设备采样率
        self.capture_sr = QComboBox()

        self.capture_sr.addItems(
            [
                "44100",
                "48000",
                "96000",
            ]
        )

        form.addRow(
            "系统录音采样率：",
            self.capture_sr,
        )

        # FFT
        self.n_fft = QComboBox()

        self.n_fft.addItems(
            [
                "512",
                "1024",
                "2048",
                "4096",
            ]
        )

        form.addRow(
            "FFT 窗口大小：",
            self.n_fft,
        )

        # Hop
        self.hop_length = QSpinBox()

        self.hop_length.setRange(
            64,
            2048,
        )

        self.hop_length.setSingleStep(
            10
        )

        form.addRow(
            "帧步长 Hop Length：",
            self.hop_length,
        )

        # Mel 维数
        self.n_mels = QSpinBox()

        self.n_mels.setRange(
            16,
            128,
        )

        form.addRow(
            "Mel 频带数量：",
            self.n_mels,
        )

        # 音量门槛
        self.energy_threshold = QDoubleSpinBox()

        self.energy_threshold.setRange(
            0.0,
            0.2,
        )

        self.energy_threshold.setDecimals(
            5
        )

        self.energy_threshold.setSingleStep(
            0.0005
        )

        form.addRow(
            "最低音量 RMS：",
            self.energy_threshold,
        )

        # 最低匹配度
        self.min_similarity = QDoubleSpinBox()

        self.min_similarity.setRange(
            0.0,
            1.0,
        )

        self.min_similarity.setDecimals(
            3
        )

        self.min_similarity.setSingleStep(
            0.01
        )

        form.addRow(
            "最低相似度：",
            self.min_similarity,
        )

        # 最低分差
        self.min_margin = QDoubleSpinBox()

        self.min_margin.setRange(
            0.0,
            1.0,
        )

        self.min_margin.setDecimals(
            3
        )

        self.min_margin.setSingleStep(
            0.005
        )

        form.addRow(
            "第一、第二名最低分差：",
            self.min_margin,
        )

        # 连续确认
        self.confirm_count = QSpinBox()

        self.confirm_count.setRange(
            1,
            5,
        )

        form.addRow(
            "连续确认次数：",
            self.confirm_count,
        )

        # 重复音效抑制时间
        self.cooldown = QDoubleSpinBox()

        self.cooldown.setRange(
            0.0,
            2.0,
        )

        self.cooldown.setDecimals(
            2
        )

        self.cooldown.setSingleStep(
            0.05
        )

        form.addRow(
            "同一音效重复抑制时间：",
            self.cooldown,
        )

        # 计算间隔
        self.analysis_interval = QDoubleSpinBox()

        self.analysis_interval.setRange(
            0.03,
            0.5,
        )

        self.analysis_interval.setDecimals(
            2
        )

        self.analysis_interval.setSingleStep(
            0.01
        )

        form.addRow(
            "分析间隔（秒）：",
            self.analysis_interval,
        )

        # 缓冲区
        self.pre_roll = QDoubleSpinBox()

        self.pre_roll.setRange(
            0.05,
            1.0,
        )

        self.pre_roll.setDecimals(
            2
        )

        form.addRow(
            "前置缓冲时间（秒）：",
            self.pre_roll,
        )

        self.post_roll = QDoubleSpinBox()

        self.post_roll.setRange(
            0.2,
            4.0,
        )

        self.post_roll.setDecimals(
            2
        )

        form.addRow(
            "后置缓冲时间（秒）：",
            self.post_roll,
        )

        self.debug_checkbox = QCheckBox(
            "显示实时候选排名和调试信息"
        )

        layout.addWidget(group)

        layout.addWidget(
            self.debug_checkbox
        )

        buttons = QHBoxLayout()

        self.save_config_btn = QPushButton(
            "保存配置"
        )

        self.save_config_btn.clicked.connect(
            self.save_config
        )

        self.reset_config_btn = QPushButton(
            "恢复默认"
        )

        self.reset_config_btn.clicked.connect(
            self.reset_config
        )

        buttons.addWidget(
            self.save_config_btn
        )

        buttons.addWidget(
            self.reset_config_btn
        )

        buttons.addStretch(1)

        layout.addLayout(
            buttons
        )

        tip = QLabel(
            "提示：降低相似度可以减少漏报，但可能增加错报；"
            "提高最低分差可加强候选区分。调整后需停止并重新启动识别。"
        )

        tip.setWordWrap(True)

        tip.setStyleSheet(
            "color: #666666;"
        )

        layout.addWidget(tip)

        layout.addStretch(1)

        return page

    def _load_config_to_widgets(self):

        cfg = {
            **DEFAULT_CONFIG,
            **self.config_data,
        }

        self.sample_rate.setCurrentText(
            str(cfg["sample_rate"])
        )

        self.capture_sr.setCurrentText(
            str(cfg["capture_sample_rate"])
        )

        self.n_fft.setCurrentText(
            str(cfg["n_fft"])
        )

        self.hop_length.setValue(
            int(cfg["hop_length"])
        )

        self.n_mels.setValue(
            int(cfg["n_mels"])
        )

        self.energy_threshold.setValue(
            float(cfg["energy_threshold"])
        )

        self.min_similarity.setValue(
            float(cfg["min_similarity"])
        )

        self.min_margin.setValue(
            float(cfg["min_margin"])
        )

        self.confirm_count.setValue(
            int(cfg["confirm_count"])
        )

        self.cooldown.setValue(
            float(cfg["cooldown_seconds"])
        )

        self.analysis_interval.setValue(
            float(cfg["analysis_interval"])
        )

        self.pre_roll.setValue(
            float(cfg["pre_roll_seconds"])
        )

        self.post_roll.setValue(
            float(cfg["post_roll_seconds"])
        )

        self.debug_checkbox.setChecked(
            bool(cfg["debug"])
        )

    def collect_config(self) -> dict:

        cfg = DEFAULT_CONFIG.copy()

        cfg.update(
            {
                "sample_rate": int(
                    self.sample_rate.currentText()
                ),

                "capture_sample_rate": int(
                    self.capture_sr.currentText()
                ),

                "n_fft": int(
                    self.n_fft.currentText()
                ),

                "hop_length": self.hop_length.value(),

                "n_mels": self.n_mels.value(),

                "energy_threshold": round(
                    self.energy_threshold.value(),
                    5,
                ),

                "min_similarity": round(
                    self.min_similarity.value(),
                    3,
                ),

                "min_margin": round(
                    self.min_margin.value(),
                    3,
                ),

                "confirm_count": self.confirm_count.value(),

                "cooldown_seconds": round(
                    self.cooldown.value(),
                    2,
                ),

                "analysis_interval": round(
                    self.analysis_interval.value(),
                    2,
                ),

                "pre_roll_seconds": round(
                    self.pre_roll.value(),
                    2,
                ),

                "post_roll_seconds": round(
                    self.post_roll.value(),
                    2,
                ),

                "debug": self.debug_checkbox.isChecked(),
            }
        )

        return cfg

    def save_config(self):

        self.config_data = self.collect_config()

        save_json(
            CONFIG_PATH,
            self.config_data,
        )

        self.statusBar().showMessage(
            f"识别参数已保存：{CONFIG_PATH}",
            5000,
        )

        self.append_log(
            "[配置] 参数已保存。"
        )

    def reset_config(self):

        self.config_data = DEFAULT_CONFIG.copy()

        self._load_default_config_widgets()

        self.save_config()

    def _load_default_config_widgets(self):

        cfg = DEFAULT_CONFIG.copy()

        self.sample_rate.setCurrentText(
            str(cfg["sample_rate"])
        )

        self.capture_sr.setCurrentText(
            str(cfg["capture_sample_rate"])
        )

        self.n_fft.setCurrentText(
            str(cfg["n_fft"])
        )

        self.hop_length.setValue(
            cfg["hop_length"]
        )

        self.n_mels.setValue(
            cfg["n_mels"]
        )

        self.energy_threshold.setValue(
            cfg["energy_threshold"]
        )

        self.min_similarity.setValue(
            cfg["min_similarity"]
        )

        self.min_margin.setValue(
            cfg["min_margin"]
        )

        self.confirm_count.setValue(
            cfg["confirm_count"]
        )

        self.cooldown.setValue(
            cfg["cooldown_seconds"]
        )

        self.analysis_interval.setValue(
            cfg["analysis_interval"]
        )

        self.pre_roll.setValue(
            cfg["pre_roll_seconds"]
        )

        self.post_roll.setValue(
            cfg["post_roll_seconds"]
        )

        self.debug_checkbox.setChecked(
            cfg["debug"]
        )

    # --------------------------------------------------------
    # 物品数据页
    # --------------------------------------------------------

    def _build_items_tab(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        left = QWidget()

        left_layout = QVBoxLayout(left)

        left_layout.addWidget(
            QLabel(
                "音频条目（data.json 的键）："
            )
        )

        self.audio_list = QListWidget()

        self.audio_list.currentRowChanged.connect(
            self._on_current_row_changed
        )

        left_layout.addWidget(
            self.audio_list
        )

        left_buttons = QHBoxLayout()

        self.add_audio_btn = QPushButton(
            "添加条目"
        )

        self.add_audio_btn.clicked.connect(
            self.add_audio_item
        )

        self.del_audio_btn = QPushButton(
            "删除条目"
        )

        self.del_audio_btn.clicked.connect(
            self.delete_audio_item
        )

        left_buttons.addWidget(
            self.add_audio_btn
        )

        left_buttons.addWidget(
            self.del_audio_btn
        )

        left_layout.addLayout(
            left_buttons
        )

        right = QWidget()

        right_layout = QVBoxLayout(right)

        right_layout.addWidget(
            QLabel(
                "格子尺寸 → 物品列表（物品名称用「、」分隔）："
            )
        )

        self.grid_table = QTableWidget(
            0,
            2,
        )

        self.grid_table.setHorizontalHeaderLabels(
            [
                "格子尺寸",
                "物品列表",
            ]
        )

        self.grid_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )

        self.grid_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )

        self.grid_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.grid_table.itemChanged.connect(
            self._on_table_changed
        )

        right_layout.addWidget(
            self.grid_table
        )

        grid_buttons = QHBoxLayout()

        self.add_grid_btn = QPushButton(
            "添加格子"
        )

        self.add_grid_btn.clicked.connect(
            self.add_grid
        )

        self.del_grid_btn = QPushButton(
            "删除格子"
        )

        self.del_grid_btn.clicked.connect(
            self.delete_grid
        )

        grid_buttons.addWidget(
            self.add_grid_btn
        )

        grid_buttons.addWidget(
            self.del_grid_btn
        )

        grid_buttons.addStretch(1)

        right_layout.addLayout(
            grid_buttons
        )

        splitter.addWidget(left)
        splitter.addWidget(right)

        splitter.setStretchFactor(
            0,
            1,
        )

        splitter.setStretchFactor(
            1,
            2,
        )

        layout.addWidget(
            splitter
        )

        bottom = QHBoxLayout()

        self.save_data_btn = QPushButton(
            "保存数据"
        )

        self.save_data_btn.clicked.connect(
            self.save_data
        )

        self.reload_data_btn = QPushButton(
            "重新加载"
        )

        self.reload_data_btn.clicked.connect(
            self.reload_data
        )

        bottom.addWidget(
            self.save_data_btn
        )

        bottom.addWidget(
            self.reload_data_btn
        )

        bottom.addStretch(1)

        layout.addLayout(bottom)

        return page

    def refresh_audio_list(
        self,
        select: str | None = None,
    ):

        self.audio_list.blockSignals(True)

        self.audio_list.clear()

        for name in self.item_data:
            self.audio_list.addItem(name)

        self.current_audio = None

        if (
            select is not None
            and select in self.item_data
        ):

            names = list(
                self.item_data.keys()
            )

            self.audio_list.setCurrentRow(
                names.index(select)
            )

            self.current_audio = select

        self.audio_list.blockSignals(False)

        self.refresh_grid_table()

    def _on_current_row_changed(
        self,
        row: int,
    ):

        self.flush_grid_table()

        if (
            0 <= row
            < self.audio_list.count()
        ):

            self.current_audio = (
                self.audio_list.item(row).text()
            )

        else:

            self.current_audio = None

        self.refresh_grid_table()

    def refresh_grid_table(self):

        self.grid_table.blockSignals(True)

        self.grid_table.setRowCount(0)

        if self.current_audio is not None:

            data = self.item_data.get(
                self.current_audio,
                {},
            )

            if not isinstance(data, dict):
                data = {}

            for grid, items in data.items():

                row = self.grid_table.rowCount()

                self.grid_table.insertRow(row)

                if isinstance(items, list):
                    text = "、".join(
                        str(item)
                        for item in items
                    )
                else:
                    text = str(items)

                self.grid_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(
                        str(grid)
                    ),
                )

                self.grid_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(
                        text
                    ),
                )

        self.grid_table.blockSignals(False)

    def flush_grid_table(self):

        audio = self.current_audio

        if audio is None:
            return

        data = {}

        for row in range(
            self.grid_table.rowCount()
        ):

            grid_item = self.grid_table.item(
                row,
                0,
            )

            items_item = self.grid_table.item(
                row,
                1,
            )

            grid = (
                grid_item.text().strip().lower()
                if grid_item
                else ""
            )

            items_text = (
                items_item.text()
                if items_item
                else ""
            )

            items = [
                item.strip()
                for item in re.split(
                    r"[、,，;；\n]+",
                    items_text,
                )
                if item.strip()
            ]

            if grid:
                data[grid] = items

        self.item_data[audio] = data

    def _on_table_changed(self, *args):

        self._mark_dirty()

    def _mark_dirty(self):

        self.dirty = True

        self._update_status()

    def add_audio_item(self):

        self.flush_grid_table()

        files = list_audio_files(
            VOICE_DIR
        )

        if not files:

            QMessageBox.information(
                self,
                "提示",
                "VoiceSource 中没有音频文件。\n"
                "请先到「音频文件」页导入。",
            )

            return

        available = [
            filename
            for filename in files
            if find_associated_key(
                filename,
                self.item_data,
            ) is None
        ]

        if not available:

            QMessageBox.information(
                self,
                "提示",
                "所有音频文件都已关联。",
            )

            return

        filename, ok = QInputDialog.getItem(
            self,
            "添加音频条目",
            "选择音频文件：",
            available,
            0,
            False,
        )

        if not ok or not filename:
            return

        self.item_data[filename] = {}

        self.refresh_audio_list(
            select=filename
        )

        self._mark_dirty()

        self.refresh_audio_files()
        self.refresh_play_list()

    def delete_audio_item(self):

        row = self.audio_list.currentRow()

        if row < 0:

            QMessageBox.information(
                self,
                "提示",
                "请先选择一个音频条目。",
            )

            return

        filename = self.audio_list.item(
            row
        ).text()

        result = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除条目「{filename}」吗？\n"
            "仅删除 data.json 中的关联，不会删除音频文件。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        self.item_data.pop(
            filename,
            None,
        )

        self.current_audio = None

        self.refresh_audio_list()

        self.refresh_audio_files()
        self.refresh_play_list()

        self._mark_dirty()

    def add_grid(self):

        if self.current_audio is None:

            QMessageBox.information(
                self,
                "提示",
                "请先选择一个音频条目。",
            )

            return

        self.flush_grid_table()

        text, ok = QInputDialog.getText(
            self,
            "添加格子",
            "请输入格子尺寸，例如 2x2、1x3：",
        )

        if not ok or not text.strip():
            return

        grid = text.strip().lower()

        if not re.fullmatch(
            r"\d+x\d+",
            grid,
        ):

            QMessageBox.warning(
                self,
                "格式错误",
                "格子尺寸应采用宽x高格式，例如 2x2。",
            )

            return

        data = self.item_data[
            self.current_audio
        ]

        if grid in data:

            QMessageBox.warning(
                self,
                "提示",
                f"格子「{grid}」已经存在。",
            )

            return

        data[grid] = []

        self.refresh_grid_table()

        self._mark_dirty()

    def delete_grid(self):

        if self.current_audio is None:
            return

        row = self.grid_table.currentRow()

        if row < 0:

            QMessageBox.information(
                self,
                "提示",
                "请先选择一行格子数据。",
            )

            return

        grid_item = self.grid_table.item(
            row,
            0,
        )

        if grid_item is None:
            return

        grid = grid_item.text().strip().lower()

        self.flush_grid_table()

        self.item_data[
            self.current_audio
        ].pop(
            grid,
            None,
        )

        self.refresh_grid_table()

        self._mark_dirty()

    def save_data(self, silent: bool = False):

        self.flush_grid_table()

        save_json(
            JSON_PATH,
            self.item_data,
        )

        self.dirty = False

        self._update_status()

        self.refresh_audio_files()
        self.refresh_play_list()

        if not silent:

            self.statusBar().showMessage(
                f"物品数据已保存：{JSON_PATH}",
                5000,
            )

    def reload_data(self):

        if self.dirty:

            result = QMessageBox.question(
                self,
                "未保存更改",
                "重新加载会丢失未保存的更改，是否继续？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )

            if result != QMessageBox.StandardButton.Yes:
                return

        self.item_data = load_json(
            JSON_PATH,
            {},
        )

        if not isinstance(self.item_data, dict):
            self.item_data = {}

        self.dirty = False
        self.current_audio = None

        self.refresh_audio_list()
        self.refresh_audio_files()
        self.refresh_play_list()

        self._update_status()

    # --------------------------------------------------------
    # 音频文件页
    # --------------------------------------------------------

    def _build_audio_tab(self):

        page = QWidget()

        layout = QVBoxLayout(page)

        layout.addWidget(
            QLabel(
                "VoiceSource 中的音频文件；"
                "带 [已关联] 表示已配置识别数据。"
            )
        )

        self.file_list = QListWidget()

        layout.addWidget(
            self.file_list
        )

        buttons = QHBoxLayout()

        self.import_btn = QPushButton(
            "导入音频"
        )

        self.import_btn.clicked.connect(
            self.import_audio
        )

        self.delete_file_btn = QPushButton(
            "删除音频"
        )

        self.delete_file_btn.clicked.connect(
            self.delete_audio_file
        )

        self.refresh_file_btn = QPushButton(
            "刷新列表"
        )

        self.refresh_file_btn.clicked.connect(
            self.refresh_audio_files
        )

        buttons.addWidget(
            self.import_btn
        )

        buttons.addWidget(
            self.delete_file_btn
        )

        buttons.addWidget(
            self.refresh_file_btn
        )

        buttons.addStretch(1)

        layout.addLayout(
            buttons
        )

        return page

    def refresh_audio_files(self):

        self.file_list.clear()

        for filename in list_audio_files(
            VOICE_DIR
        ):

            item = QListWidgetItem()

            associated = is_audio_associated(
                filename,
                self.item_data,
            )

            item.setText(
                filename
                + (
                    "  [已关联]"
                    if associated
                    else ""
                )
            )

            item.setData(
                Qt.ItemDataRole.UserRole,
                filename,
            )

            self.file_list.addItem(item)

    def import_audio(self):

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.mp3 *.mp4 *.wav *.ogg *.flac *.m4a *.aac *.wma *.opus)",
        )

        if not files:
            return

        imported = []

        for source in files:

            base = os.path.basename(source)

            destination = VOICE_DIR / base

            if (
                os.path.abspath(source)
                == os.path.abspath(destination)
            ):
                continue

            if destination.exists():

                result = QMessageBox.question(
                    self,
                    "文件已存在",
                    f"「{base}」已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                )

                if result != QMessageBox.StandardButton.Yes:
                    continue

            try:

                shutil.copy2(
                    source,
                    destination,
                )

                imported.append(base)

            except Exception as exc:

                QMessageBox.critical(
                    self,
                    "导入失败",
                    f"导入「{base}」失败：\n{exc}",
                )

        self.refresh_audio_files()

        if imported:

            self.append_log(
                f"[文件] 已导入 {len(imported)} 个音频文件。"
            )

            self.statusBar().showMessage(
                f"成功导入 {len(imported)} 个文件。",
                5000,
            )

    def delete_audio_file(self):

        item = self.file_list.currentItem()

        if item is None:

            QMessageBox.information(
                self,
                "提示",
                "请先选择一个音频文件。",
            )

            return

        filename = item.data(
            Qt.ItemDataRole.UserRole
        )

        if self.current_audio is not None:
            self.flush_grid_table()

        key = find_associated_key(
            filename,
            self.item_data,
        )

        message = (
            f"确定删除音频文件「{filename}」吗？"
        )

        if key is not None:

            message += (
                "\n该文件存在关联条目，"
                "删除文件时也会删除其 data.json 条目。"
            )

        result = QMessageBox.question(
            self,
            "确认删除",
            message,
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        try:

            (VOICE_DIR / filename).unlink()

        except OSError as exc:

            QMessageBox.critical(
                self,
                "删除失败",
                str(exc),
            )

            return

        if key is not None:

            self.item_data.pop(
                key,
                None,
            )

            save_json(
                JSON_PATH,
                self.item_data,
            )

            self.dirty = False

            self.current_audio = None

            self.refresh_audio_list()

        self.refresh_audio_files()
        self.refresh_play_list()
        self._update_status()

    # --------------------------------------------------------
    # 识别线程控制
    # --------------------------------------------------------

    def on_run_clicked(self):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.stop_recognition()

        else:

            self.start_recognition()

    def start_recognition(self):

        # 防止保存配置时仍处于旧参数状态
        self.save_config()

        self.flush_grid_table()

        if self.dirty:
            self.save_data(silent=True)

        if not self.item_data:

            QMessageBox.warning(
                self,
                "无法启动",
                "data.json 为空。\n"
                "请到「物品数据」页添加音频条目。",
            )

            return

        self.worker = RecognitionWorker(
            self.collect_config(),
            self.item_data.copy(),
        )

        self.worker.log.connect(
            self.append_log
        )

        self.worker.finished.connect(
            self.on_recognition_finished
        )

        self.worker.start()

        self.run_button.setText(
            "停止识别"
        )

        self.run_state_label.setText(
            "当前状态：正在识别"
        )

        self.append_log(
            "══════════ 开始识别 ══════════"
        )

    def stop_recognition(self):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            self.worker.stop()

            self.run_button.setEnabled(
                False
            )

            self.run_state_label.setText(
                "当前状态：正在停止"
            )

            self.append_log(
                "[状态] 已发送停止信号，正在关闭音频采集。"
            )

    def on_recognition_finished(self):

        self.run_button.setText(
            "开始识别"
        )

        self.run_button.setEnabled(
            True
        )

        self.run_state_label.setText(
            "当前状态：未启动"
        )

    # --------------------------------------------------------
    # 运行页音频试听
    # --------------------------------------------------------

    def refresh_play_list(self):

        if (
            self.player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.player.stop()

        self._playing_button = None
        self._playing_file = None

        self.play_list.clear()

        for name, data in self.item_data.items():

            item = QListWidgetItem()

            widget = self._make_play_row(
                name,
                data,
            )

            item.setSizeHint(
                widget.sizeHint()
            )

            self.play_list.addItem(item)

            self.play_list.setItemWidget(
                item,
                widget,
            )

    def _make_play_row(
        self,
        name: str,
        data: Any,
    ) -> QWidget:

        row = QWidget()

        horizontal = QHBoxLayout(row)

        horizontal.setContentsMargins(
            4,
            6,
            4,
            6,
        )

        play_button = QPushButton(
            "▶"
        )

        play_button.setFixedSize(
            36,
            30,
        )

        play_button.setToolTip(
            f"播放 {name}"
        )

        play_button.clicked.connect(
            lambda checked=False,
            filename=name,
            button=play_button:
                self.toggle_play(
                    filename,
                    button,
                )
        )

        horizontal.addWidget(
            play_button,
            0,
            Qt.AlignmentFlag.AlignTop,
        )

        vertical = QVBoxLayout()

        vertical.setSpacing(3)

        name_label = QLabel(
            name
        )

        font = name_label.font()
        font.setBold(True)

        name_label.setFont(
            font
        )

        vertical.addWidget(
            name_label
        )

        lines = []

        if isinstance(data, dict):

            for grid, items in data.items():

                if isinstance(items, list):
                    text = "、".join(
                        str(item)
                        for item in items
                    )
                else:
                    text = str(items)

                lines.append(
                    f"{grid} → {text}"
                )

        data_label = QLabel(
            "\n".join(lines)
            if lines
            else "（暂无关联数据）"
        )

        data_label.setWordWrap(True)

        data_label.setStyleSheet(
            "color: #666666;"
        )

        vertical.addWidget(
            data_label
        )

        horizontal.addLayout(
            vertical,
            1,
        )

        return row

    def _reset_play_state(self):
        """统一重置试听按钮及当前播放文件状态。"""

        if self._playing_button is not None:
            self._playing_button.setText("▶")

        self._playing_button = None
        self._playing_file = None

    def toggle_play(
            self,
            filename: str,
            button: QPushButton,
    ):

        # 只有同一个文件确实正在播放时，再次点击才执行停止
        is_playing = (
                self.player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState
        )

        if self._playing_file == filename and is_playing:
            self.player.stop()
            self._reset_play_state()
            return

        # 文件不存在时直接提示
        path = resolve_audio_path(filename)

        if path is None:
            QMessageBox.warning(
                self,
                "提示",
                f"找不到音频文件：{filename}",
            )
            return

        # 恢复上一个试听按钮的显示状态
        if self._playing_button is not None:
            self._playing_button.setText("▶")

        # 停止旧音频并清理旧状态
        self.player.stop()
        self._reset_play_state()

        # 先记录当前文件和按钮，再启动播放
        self._playing_button = button
        self._playing_file = filename
        button.setText("⏸")

        try:
            self.player.setSource(
                QUrl.fromLocalFile(str(path.resolve()))
            )

            # 等待媒体加载完成后，再从头播放
            self.player.setPosition(0)
            self.player.play()


        except Exception as exc:
            self._reset_play_state()
            self.append_log(
                f"[播放错误] {filename}: "
                f"{type(exc).__name__}: {exc}"
            )



    def _on_media_status_changed(self, status):

        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._reset_play_state()

        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            filename = self._playing_file or "未知文件"
            self._reset_play_state()
            self.append_log(
                f"[播放错误] 媒体无效：{filename}"
            )

    def _on_player_error(self, error, error_string):
        """处理播放器错误并恢复试听状态。"""

        filename = self._playing_file or "未知文件"

        self._reset_play_state()

        self.append_log(
            f"[播放错误] {filename} | "
            f"{error.name if hasattr(error, 'name') else error} | "
            f"{error_string}"
        )

    # --------------------------------------------------------
    # 日志和状态
    # --------------------------------------------------------

    def append_log(
        self,
        text: str,
    ):

        self.log_view.appendPlainText(
            text
        )

        scrollbar = (
            self.log_view.verticalScrollBar()
        )

        scrollbar.setValue(
            scrollbar.maximum()
        )

    def clear_log(self):

        self.log_view.clear()

    def _update_status(self):

        message = (
            f"数据文件：{JSON_PATH}"
        )

        if self.dirty:
            message += "    ● 有未保存的修改"

        self.statusBar().showMessage(
            message
        )

    def _on_tab_changed(
        self,
        index: int,
    ):

        # 运行页是第一个标签页
        if index == 0:
            self.refresh_play_list()

    # --------------------------------------------------------
    # 程序退出
    # --------------------------------------------------------

    def closeEvent(
        self,
        event,
    ):

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):

            result = QMessageBox.question(
                self,
                "确认退出",
                "识别正在运行，确定停止并退出吗？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )

            if result != QMessageBox.StandardButton.Yes:

                event.ignore()
                return

            self.worker.stop()

            if not self.worker.wait(4000):

                QMessageBox.information(
                    self,
                    "正在停止",
                    "音频采集线程尚未完全退出，请稍后再关闭窗口。",
                )

                event.ignore()
                return

        if self.dirty:

            result = QMessageBox.warning(
                self,
                "未保存的数据",
                "物品数据有未保存的修改，是否保存？",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if result == QMessageBox.StandardButton.Save:

                self.save_data(
                    silent=True
                )

            elif result == QMessageBox.StandardButton.Cancel:

                event.ignore()
                return

        event.accept()


# ============================================================
# 7. 启动程序
# ============================================================

def main():

    VOICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    app = QApplication(
        sys.argv
    )

    app.setFont(
        QFont(
            "Microsoft YaHei UI",
            9,
        )
    )

    app.setStyle(
        "Fusion"
    )

    window = MainWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()