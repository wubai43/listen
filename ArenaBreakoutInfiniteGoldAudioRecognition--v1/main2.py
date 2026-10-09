import os
import sys
import json
import time
import warnings
from collections import Counter
import numpy as np
import scipy.signal as signal
import librosa
import soundcard as sc

warnings.filterwarnings("ignore")

# ================= 1. 全局配置与参数设置 =================
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_PATH = os.path.join(BASE_DIR, "VoiceSource", "data.json")
VOICE_DIR = os.path.join(BASE_DIR, "VoiceSource")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# 默认值回调到宽松档（接近原版），所有阈值可在 config.json 里再调
DEFAULT_CONFIG = {
    "sample_rate": 22050,        # 统一采样率
    "fft_window_size": 2048,     # FFT 窗口大小
    "hop_size": 512,             # 帧步长
    "energy_threshold": 0.005,   # 激活音量门槛
    "cooldown_time": 0.7,        # 输出后的冷却时间
    "peak_threshold": 0.15,      # 频谱峰值阈值（0.18~0.25 可调）
    "min_score": 8,              # 最低匹配得分（3~8 可调）
    "min_ratio": 0.40,           # 最低匹配率（0.22~0.45 可调）
    "confirm_blocks": 3,         # 连续确认块数（1 = 立即输出）
    "debug": True,               # 调试日志开关
}

SAMPLE_RATE = DEFAULT_CONFIG["sample_rate"]
FFT_WINDOW_SIZE = DEFAULT_CONFIG["fft_window_size"]
HOP_SIZE = DEFAULT_CONFIG["hop_size"]
ENERGY_THRESHOLD = DEFAULT_CONFIG["energy_threshold"]
COOLDOWN_TIME = DEFAULT_CONFIG["cooldown_time"]
PEAK_THRESHOLD = DEFAULT_CONFIG["peak_threshold"]
MIN_SCORE = DEFAULT_CONFIG["min_score"]
MIN_RATIO = DEFAULT_CONFIG["min_ratio"]
CONFIRM_BLOCKS = DEFAULT_CONFIG["confirm_blocks"]
DEBUG = DEFAULT_CONFIG["debug"]


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"⚠️  读取 config.json 失败，使用默认配置: {e}")
    return cfg


def apply_config(cfg):
    global SAMPLE_RATE, FFT_WINDOW_SIZE, HOP_SIZE, ENERGY_THRESHOLD
    global COOLDOWN_TIME, PEAK_THRESHOLD, MIN_SCORE, MIN_RATIO, CONFIRM_BLOCKS, DEBUG
    SAMPLE_RATE = int(cfg.get("sample_rate", DEFAULT_CONFIG["sample_rate"]))
    FFT_WINDOW_SIZE = int(cfg.get("fft_window_size", DEFAULT_CONFIG["fft_window_size"]))
    HOP_SIZE = int(cfg.get("hop_size", DEFAULT_CONFIG["hop_size"]))
    ENERGY_THRESHOLD = float(cfg.get("energy_threshold", DEFAULT_CONFIG["energy_threshold"]))
    COOLDOWN_TIME = float(cfg.get("cooldown_time", DEFAULT_CONFIG["cooldown_time"]))
    PEAK_THRESHOLD = float(cfg.get("peak_threshold", DEFAULT_CONFIG["peak_threshold"]))
    MIN_SCORE = int(cfg.get("min_score", DEFAULT_CONFIG["min_score"]))
    MIN_RATIO = float(cfg.get("min_ratio", DEFAULT_CONFIG["min_ratio"]))
    CONFIRM_BLOCKS = int(cfg.get("confirm_blocks", DEFAULT_CONFIG["confirm_blocks"]))
    DEBUG = bool(cfg.get("debug", DEFAULT_CONFIG["debug"]))


apply_config(load_config())

# ================= 2. 音频指纹提取算法 =================
def extract_fingerprints(samples, sr=SAMPLE_RATE):
    """提取携带相对时间戳的频域峰值指纹（按帧归一化版本）"""
    if len(samples) < FFT_WINDOW_SIZE:
        return {}

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    _, _, spec = signal.stft(
        samples,
        fs=sr,
        nperseg=FFT_WINDOW_SIZE,
        noverlap=FFT_WINDOW_SIZE - HOP_SIZE
    )
    spec_mag = np.abs(spec)

    # 按帧（列）归一化：每帧最强峰 = 1.0
    frame_max = spec_mag.max(axis=0, keepdims=True)
    spec_mag = spec_mag / np.maximum(frame_max, 1e-10)

    hashes = {}
    num_freq_bins, num_time_bins = spec_mag.shape
    bands = [(0, 12), (12, 28), (28, 50), (50, 90)]

    for t in range(num_time_bins):
        # 整帧都很弱的帧（静音帧/纯底噪帧）直接跳过
        if frame_max[0, t] < 1e-6:
            continue

        for band_idx, (f_min, f_max) in enumerate(bands):
            if f_max < num_freq_bins:
                sub_spec = spec_mag[f_min:f_max, t]
                max_f = np.argmax(sub_spec) + f_min
                max_v = spec_mag[max_f, t]

                if max_v > PEAK_THRESHOLD:
                    key = (band_idx, max_f)  # 复合键：频段+频率
                    hashes[key] = hashes.get(key, []) + [t]

    return hashes

# ================= 3. 构建模板指纹库 =================
def build_target_fingerprints(item_data, voice_dir):
    print("--------------------------------------------------")
    print("正在预加载并分析 VoiceSource 模板音频指纹...")
    target_fps = {}

    for file_name in item_data.keys():
        file_path = os.path.join(voice_dir, file_name)

        if not os.path.exists(file_path):
            alt_ext = ".mp3" if file_name.endswith(".mp4") else ".mp4"
            alt_path = os.path.join(voice_dir, os.path.splitext(file_name)[0] + alt_ext)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                print(f"⚠️  警告: 找不到文件 -> {file_path}")
                continue

        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

            all_fps = []
            max_fp_count = 0

            for speed in [0.95, 1.0, 1.05]:
                if speed == 1.0:
                    y_stretched = y
                else:
                    y_stretched = librosa.effects.time_stretch(y, rate=speed)

                fps = extract_fingerprints(y_stretched, sr)
                all_fps.append(fps)
                max_fp_count = max(max_fp_count, len(fps))

            target_fps[file_name] = {
                "fps_list": all_fps,
                "total_features": max_fp_count
            }
            print(f"✅ 成功加载: {file_name:<10} | 基础特征数: {max_fp_count}")

        except Exception as e:
            print(f"❌ 加载文件 {file_name} 失败: {e}")

    print(f"模板库加载完成，共有效加载 {len(target_fps)} 个文件。")
    print("--------------------------------------------------\n")
    return target_fps

# ================= 4. 时序对齐匹配逻辑 =================
def match_fingerprints(current_hashes, target_fps_list):
    best_score = 0

    for ref_hashes in target_fps_list:
        offsets = []
        for freq in current_hashes:
            if freq in ref_hashes:
                for t_curr in current_hashes[freq]:
                    for t_ref in ref_hashes[freq]:
                        offsets.append(t_curr - t_ref)

        if offsets:
            offset_counts = Counter(offsets)
            _, count = offset_counts.most_common(1)[0]
            if count > best_score:
                best_score = count

    return best_score

# ================= 5. 即时输出匹配引擎（带全程调试日志） =================
def start_realtime_listener(item_data, target_fingerprints, stop_event=None):
    try:
        speaker = sc.default_speaker()
        mics = sc.all_microphones(include_loopback=True)

        loopback_mic = None
        for mic in mics:
            if mic.isloopback and speaker.name in mic.name:
                loopback_mic = mic
                break

        if loopback_mic is None:
            for mic in mics:
                if mic.isloopback:
                    loopback_mic = mic
                    break

        if loopback_mic is None:
            print("❌ 错误: 未能在 Windows 系统中检测到任何 Loopback 设备。")
            return

    except Exception as e:
        print(f"❌ 初始化内录设备失败: {e}")
        return

    print(f"🎧 成功锁定内录通道 -> [{loopback_mic.name}]")

    buffer_duration = 2.0
    buffer_size = int(SAMPLE_RATE * buffer_duration)
    ring_buffer = np.zeros(buffer_size, dtype=np.float32)

    last_output_time = 0
    record_sr = 48000
    block_size = int(record_sr * 0.1)

    confirm_name = None
    confirm_count = 0
    log_tick = 0   # 用于限制日志刷屏频率

    print(f"⚙️  当前参数: 峰值阈值={PEAK_THRESHOLD} | min_score={MIN_SCORE} | min_ratio={MIN_RATIO} | 激活门槛={ENERGY_THRESHOLD} | 冷却={COOLDOWN_TIME}s")
    print("🚀 匹配引擎已就绪，正在监听系统音频...\n")

    with loopback_mic.recorder(samplerate=record_sr, channels=1) as recorder:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("⏹️ 已手动停止监听。")
                break
            current_time = time.time()
            data = recorder.record(numframes=block_size)
            data_np = data.squeeze()

            if record_sr != SAMPLE_RATE:
                data_np = librosa.resample(data_np, orig_sr=record_sr, target_sr=SAMPLE_RATE)

            chunk_len = len(data_np)
            if chunk_len > 0:
                ring_buffer = np.roll(ring_buffer, -chunk_len)
                ring_buffer[-chunk_len:] = data_np

            rms_energy = np.sqrt(np.mean(ring_buffer ** 2))
            log_tick += 1

            matched_now = False

            # --- 日志：冷却期中不识别（低频打印，避免刷屏） ---
            in_cooldown = current_time - last_output_time <= COOLDOWN_TIME
            if DEBUG and in_cooldown and log_tick % 25 == 0:
                print(f"⏳ [冷却中 {COOLDOWN_TIME - (current_time - last_output_time):.1f}s] 音量={rms_energy:.4f}")

            if not in_cooldown:
                if rms_energy <= ENERGY_THRESHOLD:
                    # --- 日志：音量不足被跳过（低频打印） ---
                    if DEBUG and log_tick % 25 == 0:
                        print(f"🔇 [音量不足] RMS={rms_energy:.4f} < 门槛={ENERGY_THRESHOLD}（提高 energy_threshold 或检查内录音量）")
                else:
                    current_hashes = extract_fingerprints(ring_buffer, SAMPLE_RATE)

                    if len(current_hashes) == 0:
                        # --- 日志：有声音但提不出任何峰值 ---
                        if DEBUG:
                            print(f"🔇 [无有效峰值] 音量={rms_energy:.4f}，但没有任何频率超过峰值阈值 {PEAK_THRESHOLD}（可调低 peak_threshold）")
                    else:
                        best_match = None
                        max_ratio = 0.0
                        matched_score = 0
                        candidate_log = []

                        for audio_name, target_info in target_fingerprints.items():
                            score = match_fingerprints(current_hashes, target_info["fps_list"])
                            total_features = target_info["total_features"]
                            ratio = score / total_features if total_features > 0 else 0
                            candidate_log.append(f"{audio_name}: score={score}, ratio={ratio:.2f}")

                            if score >= MIN_SCORE and ratio >= MIN_RATIO and ratio > max_ratio:
                                max_ratio = ratio
                                matched_score = score
                                best_match = audio_name

                        if best_match is None:
                            # --- 日志：有峰值但没有任何模板达标（最关键的诊断信息） ---
                            if DEBUG:
                                print(f"❌ [未命中] 当前峰值数={len(current_hashes)}，各模板得分 -> " + " | ".join(candidate_log))
                                print(f"   提示: 需 score>={MIN_SCORE} 且 ratio>={MIN_RATIO}，可调低 min_score/min_ratio")
                        else:
                            matched_now = True
                            if DEBUG and CONFIRM_BLOCKS > 1:
                                print(f"🟡 [候选] {best_match} (score={matched_score}, ratio={max_ratio:.2f}) 连续确认 {confirm_count+1}/{CONFIRM_BLOCKS}")

                            if best_match == confirm_name:
                                confirm_count += 1
                            else:
                                confirm_name = best_match
                                confirm_count = 1

                            if confirm_count >= CONFIRM_BLOCKS:
                                print("=" * 50)
                                print(f"🎯 识别成功 | 音频: {best_match} (得分: {matched_score} | 匹配度: {max_ratio*100:.1f}% | 音量: {rms_energy:.4f})")
                                print(f"📦 关联数据: {json.dumps(item_data[best_match], ensure_ascii=False)}")
                                print("=" * 50 + "\n")

                                ring_buffer.fill(0)
                                last_output_time = current_time
                                confirm_name = None
                                confirm_count = 0

            if not matched_now:
                confirm_name = None
                confirm_count = 0

            time.sleep(0.02)

# ================= 6. 主程序入口 =================
if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"❌ 错误: 找不到 json 配置文件 ({JSON_PATH})")
        exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        item_data = json.load(f)

    target_fingerprints = build_target_fingerprints(item_data, VOICE_DIR)

    if not target_fingerprints:
        print("❌ 未生成有效指纹，程序退出。")
        exit(1)

    try:
        start_realtime_listener(item_data, target_fingerprints)
    except KeyboardInterrupt:
        print("\n已手动停止监听。")