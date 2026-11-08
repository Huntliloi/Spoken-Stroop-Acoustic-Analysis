import argparse
import os
import re
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# User configuration
# =========================
AUDIO_ROOT = str(Path(__file__).resolve().parents[2] / "data" / "example" / "audio")
OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "results" / "recomputed" / "example_asr")

# Use local model folders if you have downloaded them.
ASR_MODEL = os.environ.get("QWEN_ASR_MODEL", "")
ALIGNER_MODEL = os.environ.get("QWEN_ALIGNER_MODEL", "")

# If your torch is CPU-only, keep DEVICE="cpu".
# If you have CUDA PyTorch installed, set DEVICE="cuda:0".
DEVICE = "cpu"
DTYPE = "float32"  # "float32", "float16", "bfloat16", or "auto"
LANGUAGE = "Chinese"
USE_GUI_FOLDER_PICKER = False
LOCAL_FILES_ONLY = True
EXPECTED_TOKENS_PER_TASK = 12

# =========================
# Boundary refinement config
# Processing step.
# =========================
ENERGY_FRAME_MS = 6.0
ENERGY_HOP_MS = 1.0
ENERGY_SMOOTH_MS = 5.0

PAIR_CONTEXT_MIN_PAD_SEC = 0.08
PAIR_CONTEXT_EXTRA_GAP_RATIO = 0.80
PAIR_CONTEXT_DUR_RATIO = 0.40
PAIR_CONTEXT_MAX_SEC = 3.00

MIN_SILENCE_DUR_FLOOR_SEC = 0.025
MIN_SILENCE_DUR_CEIL_SEC = 0.60

SPEECH_SCORE_LOW_Q = 15
SPEECH_SCORE_HIGH_Q = 70
SPEECH_SCORE_THR_RATIO = 0.20

# Processing step.
WEIGHT_ENERGY = 0.40
WEIGHT_ENV = 0.30
WEIGHT_FLUCT = 0.30

# =========================
# Optional GUI picker
# =========================
if USE_GUI_FOLDER_PICKER:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    chosen_audio = filedialog.askdirectory(title="请选择音频根目录")
    if chosen_audio:
        AUDIO_ROOT = chosen_audio
    chosen_output = filedialog.askdirectory(title="请选择输出目录")
    if chosen_output:
        OUTPUT_DIR = chosen_output

# =========================
# Imports after config
# =========================
try:
    import torch
except Exception as e:
    raise ImportError("未能导入 torch。请先正确安装 PyTorch。") from e

try:
    import soundfile as sf
except Exception as e:
    raise ImportError("未能导入 soundfile。请先安装: pip install soundfile") from e

try:
    from qwen_asr import Qwen3ASRModel
except Exception as e:
    raise ImportError("未能导入 qwen_asr。请先在当前环境安装 qwen-asr。") from e


# =========================
# Utilities
# =========================
COLORS = ["红", "黄", "蓝", "绿"]
COLOR_SET = set(COLORS)
FILLERS = {
    "嗯", "呃", "额", "啊", "唉", "欸", "诶", "哦", "奥", "欧", "哎", "恩", "唔", "嗯嗯"
}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("检测到当前 PyTorch 不支持 CUDA，自动切换到 CPU。")
        return "cpu"
    return device


def resolve_dtype(device: str, dtype: str):
    dtype = (dtype or "auto").lower()
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype, torch.float32)


def task_from_filename(name: str) -> str:
    lower = name.lower()
    if "_hanzi" in lower or "_stroop_w" in lower:
        return "hanzi"
    if "_dots" in lower or "_stroop_c." in lower:
        return "dots"
    if "_stt" in lower or "_stroop_cw" in lower:
        return "STT"
    return "unknown"


def subject_id_from_folder(folder_name: str) -> str:
    return folder_name[-11:] if len(folder_name) >= 11 else folder_name


def detect_task_from_filename(filename: str) -> Optional[str]:
    """Process spoken Stroop recordings."""
    name = filename.lower()

    if "_stroop_hanzi" in name or name.endswith("hanzi.wav") or name.endswith("_stroop_w.wav"):
        return "hanzi"
    if "_stroop_dots" in name or name.endswith("dots.wav") or name.endswith("_stroop_c.wav"):
        return "dots"
    if "_stroop_stt" in name or name.endswith("stt.wav") or name.endswith("_stroop_cw.wav"):
        return "STT"

    return None


def find_audio_files(audio_root: str) -> List[Path]:
    """Process spoken Stroop recordings."""
    root = Path(audio_root)
    if not root.exists():
        raise FileNotFoundError(f"音频根目录不存在: {audio_root}")

    files: List[Path] = []
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        stroop_dir = subj_dir / "session_1" / "04_Stroop"
        if not stroop_dir.is_dir():
            stroop_dir = subj_dir / "session_1"
        if not stroop_dir.is_dir():
            continue
        for wav_path in sorted(stroop_dir.glob("*.wav")):
            name = wav_path.name.lower()
            if "stroop" not in name:
                continue
            task = detect_task_from_filename(wav_path.name)
            if task in {"hanzi", "dots", "STT"}:
                files.append(wav_path)
    return files


# =========================
# Model loading / inference
# =========================
def load_model(asr_model_name: str, aligner_model_name: str, device: str, dtype: str):
    device = normalize_device(device)
    torch_dtype = resolve_dtype(device, dtype)

    kwargs: Dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
        "forced_aligner": aligner_model_name,
        "forced_aligner_kwargs": {
            "device_map": device,
            "dtype": torch_dtype,
        },
        "trust_remote_code": True,
        "local_files_only": LOCAL_FILES_ONLY,
    }

    print("音频根目录:", AUDIO_ROOT)
    print("输出目录:", OUTPUT_DIR)
    print("ASR 模型:", asr_model_name)
    print("对齐模型:", aligner_model_name)
    print("设备:", device)
    print("精度:", dtype)
    print("正在加载模型，请稍候...")
    model = Qwen3ASRModel.from_pretrained(asr_model_name, **kwargs)
    print("模型加载完成。")
    return model


def serialize_timestamp_items(items: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if items is None:
        return rows
    for item in items:
        if isinstance(item, dict):
            rows.append({
                "text": item.get("text", ""),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
            })
        else:
            rows.append({
                "text": getattr(item, "text", ""),
                "start_time": getattr(item, "start_time", None),
                "end_time": getattr(item, "end_time", None),
            })
    return rows


def call_asr(model, audio_path: str) -> Dict[str, Any]:
    results = model.transcribe(
        audio=audio_path,
        language=LANGUAGE,
        return_time_stamps=True,
    )

    if not results:
        return {
            "language": None,
            "text": "",
            "time_stamps": [],
            "raw_result": None,
        }

    r = results[0]
    text = getattr(r, "text", "") or ""
    language = getattr(r, "language", None)
    time_stamps = serialize_timestamp_items(getattr(r, "time_stamps", None) or [])

    return {
        "language": language,
        "text": text,
        "time_stamps": time_stamps,
        "raw_result": r,
    }


# =========================
# Text cleaning
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[，,。！？!?.、；;：:\-—_/\\|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_color_candidates_from_text(text: str) -> List[str]:
    if not text:
        return []
    candidates: List[str] = []
    for ch in text:
        if ch in COLOR_SET or ch in FILLERS:
            candidates.append(ch)
    return candidates


def remove_fillers(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in FILLERS]


def clean_color_sequence(raw_tokens: List[str], expected_n: int = EXPECTED_TOKENS_PER_TASK) -> Tuple[List[str], str]:
    """Process spoken Stroop recordings."""
    tokens = remove_fillers(raw_tokens)
    tokens = [t for t in tokens if t in COLOR_SET]

    reason = ""
    if len(tokens) < expected_n:
        reason = f"too_few_tokens:{len(tokens)}"
    elif len(tokens) > expected_n:
        tokens = list(tokens)
        reason = f"surplus_tokens_retained:{len(tokens)}"

    return tokens, reason


def filter_aligned_items_to_colors(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        txt = str(item.get("text", "") or "").strip()
        if len(txt) == 1 and txt in COLOR_SET:
            out.append(item)
        else:
            for ch in txt:
                if ch in COLOR_SET:
                    out.append({
                        "text": ch,
                        "start_time": item.get("start_time"),
                        "end_time": item.get("end_time"),
                    })
    return out


def align_cleaned_text_if_needed(model, audio_path: str, cleaned_colors: List[str]) -> List[Dict[str, Any]]:
    """Process spoken Stroop recordings."""
    if not cleaned_colors:
        return []

    aligner = getattr(model, "forced_aligner", None)
    if aligner is None:
        return []

    text = "".join(cleaned_colors)
    try:
        results = aligner.align(audio=audio_path, text=text, language=LANGUAGE)
        if not results:
            return []
        return serialize_timestamp_items(results[0])
    except Exception:
        return []


# =========================
# Boundary refinement features
# =========================
def load_audio_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(audio_path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    return y, sr


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x.astype(np.float32, copy=False)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x.astype(np.float32, copy=False), (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32, copy=False)


def moving_std(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return np.zeros_like(x, dtype=np.float32)
    if k % 2 == 0:
        k += 1
    x = x.astype(np.float32, copy=False)
    mean_x = moving_average(x, k)
    mean_x2 = moving_average(x * x, k)
    var = np.maximum(mean_x2 - mean_x * mean_x, 0.0)
    return np.sqrt(var).astype(np.float32)


def robust_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -8.0, 8.0).astype(np.float32)


def compute_boundary_features(
    y: np.ndarray,
    sr: int,
    frame_ms: float = ENERGY_FRAME_MS,
    hop_ms: float = ENERGY_HOP_MS,
    smooth_ms: float = ENERGY_SMOOTH_MS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = max(16, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))

    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))

    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10).astype(np.float32)
    log_energy = np.log(rms + 1e-8).astype(np.float32)

    env = np.mean(np.abs(frames), axis=1).astype(np.float32)

    smooth_k = max(1, int(round(smooth_ms / hop_ms)))
    env_smooth = moving_average(env, smooth_k)
    fluct = moving_std(env_smooth, max(3, smooth_k))

    log_energy = moving_average(log_energy, smooth_k)
    times = ((np.arange(len(log_energy), dtype=np.float32) * hop) + frame / 2.0) / float(sr)

    return times, log_energy, env_smooth, fluct


def find_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if len(mask) == 0:
        return runs

    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False

    if in_run:
        runs.append((start, len(mask) - 1))
    return runs

def merge_close_runs(
    runs: List[Tuple[int, int]],
    seg_t: np.ndarray,
    max_bridge_sec: float,
) -> List[Tuple[int, int]]:
    if not runs:
        return runs

    merged = [[runs[0][0], runs[0][1]]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap_sec = float(seg_t[s] - seg_t[prev_e])
        if gap_sec <= max_bridge_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(int(s), int(e)) for s, e in merged]
def refine_one_pair_boundary(
    times: np.ndarray,
    log_energy: np.ndarray,
    env: np.ndarray,
    fluct: np.ndarray,
    prev_item: Dict[str, Any],
    next_item: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """Process spoken Stroop recordings."""
    if prev_item.get("start_time") is None or prev_item.get("end_time") is None:
        return None
    if next_item.get("start_time") is None or next_item.get("end_time") is None:
        return None

    prev_start = float(prev_item["start_time"])
    prev_end = float(prev_item["end_time"])
    next_start = float(next_item["start_time"])
    next_end = float(next_item["end_time"])

    if next_end <= prev_start:
        return None

    prev_dur = max(1e-3, prev_end - prev_start)
    next_dur = max(1e-3, next_end - next_start)
    gap = max(0.0, next_start - prev_end)

    pad = max(
        PAIR_CONTEXT_MIN_PAD_SEC,
        PAIR_CONTEXT_EXTRA_GAP_RATIO * gap,
        PAIR_CONTEXT_DUR_RATIO * max(prev_dur, next_dur),
    )
    pad = min(pad, PAIR_CONTEXT_MAX_SEC)

    search_left = max(prev_start, prev_end - pad)
    search_right = min(next_end, next_start + pad)

    if search_right <= search_left:
        return None

    li = int(np.searchsorted(times, search_left, side="left"))
    ri = int(np.searchsorted(times, search_right, side="right")) - 1

    li = max(0, min(li, len(times) - 1))
    ri = max(0, min(ri, len(times) - 1))

    if ri - li < 6:
        return None

    seg_t = times[li:ri + 1]
    seg_e = log_energy[li:ri + 1]
    seg_env = env[li:ri + 1]
    seg_fluct = fluct[li:ri + 1]

    z_e = robust_scale(seg_e)
    z_env = robust_scale(seg_env)
    z_fluct = robust_scale(seg_fluct)

    speech_score = (
        WEIGHT_ENERGY * z_e +
        WEIGHT_ENV * z_env +
        WEIGHT_FLUCT * z_fluct
    ).astype(np.float32)
    speech_score = moving_average(speech_score, 3)

    q_low = float(np.percentile(speech_score, SPEECH_SCORE_LOW_Q))
    q_high = float(np.percentile(speech_score, SPEECH_SCORE_HIGH_Q))
    score_thr = q_low + SPEECH_SCORE_THR_RATIO * max(1e-6, (q_high - q_low))

    silence_mask = speech_score <= score_thr
    runs = find_true_runs(silence_mask)

    boundary_mid = 0.5 * (prev_end + next_start)
    window_width = max(1e-6, search_right - search_left)

    # Processing step.
    bridge_sec = min(0.12, max(0.04, 0.10 * gap))
    runs = merge_close_runs(runs, seg_t, bridge_sec)

    min_sil_dur = max(
        MIN_SILENCE_DUR_FLOOR_SEC,
        min(MIN_SILENCE_DUR_CEIL_SEC, 0.25 * gap)
    )
    target_sil_dur = max(0.05, min(1.20, 0.85 * gap))
    # Processing step.
    mid_runs = []
    for s, e in runs:
        run_start_t = float(seg_t[s])
        run_end_t = float(seg_t[e])
        run_dur = max(0.0, run_end_t - run_start_t)

        if run_dur < min_sil_dur:
            continue

        if run_start_t <= boundary_mid <= run_end_t:
            mid_runs.append((s, e, run_dur))

    if mid_runs:
        s, e, _ = max(mid_runs, key=lambda x: x[2])

        new_prev_end = float(seg_t[s])
        new_next_start = float(seg_t[e])

        new_prev_end = max(prev_start, min(new_prev_end, next_end))
        new_next_start = max(prev_start, min(new_next_start, next_end))

        if new_next_start < new_prev_end:
            mid = 0.5 * (new_prev_end + new_next_start)
            new_prev_end = mid
            new_next_start = mid

        return round(new_prev_end, 4), round(new_next_start, 4)
    best_run = None
    best_score = None

    for s, e in runs:
        run_start_t = float(seg_t[s])
        run_end_t = float(seg_t[e])
        run_dur = max(0.0, run_end_t - run_start_t)

        if run_dur < min_sil_dur:
            continue

        run_mid = 0.5 * (run_start_t + run_end_t)
        dist_norm = abs(run_mid - boundary_mid) / window_width
        dur_norm = min(run_dur / max(target_sil_dur, 1e-6), 1.5)
        dur_diff_norm = abs(run_dur - target_sil_dur) / max(target_sil_dur, 1e-6)

        run_score_mean = float(np.mean(speech_score[s:e + 1]))

        before = speech_score[max(0, s - 4):s]
        after = speech_score[e + 1:min(len(speech_score), e + 5)]
        before_mean = float(np.mean(before)) if len(before) > 0 else run_score_mean
        after_mean = float(np.mean(after)) if len(after) > 0 else run_score_mean
        contrast = 0.5 * ((before_mean - run_score_mean) + (after_mean - run_score_mean))

        contains_mid_bonus = 0.9 if (run_start_t <= boundary_mid <= run_end_t) else 0.0

        score = (
            contains_mid_bonus
            + 0.8 * contrast
            - 1.4 * dist_norm
            + 0.10 * dur_norm
            - 0.05 * dur_diff_norm
        )

        if (best_score is None) or (score > best_score):
            best_score = score
            best_run = (s, e)

    if best_run is not None:
        s, e = best_run
        new_prev_end = float(seg_t[s])
        new_next_start = float(seg_t[e])

        new_prev_end = max(prev_start, min(new_prev_end, next_end))
        new_next_start = max(prev_start, min(new_next_start, next_end))

        if new_next_start < new_prev_end:
            mid = 0.5 * (new_prev_end + new_next_start)
            new_prev_end = mid
            new_next_start = mid

        return round(new_prev_end, 4), round(new_next_start, 4)

    # Processing step.
    m = int(np.argmin(speech_score))
    valley_min = float(speech_score[m])
    valley_ref = float(np.percentile(speech_score, 25))
    basin_level = valley_min + 0.35 * max(1e-6, valley_ref - valley_min)

    l = m
    while l > 0 and speech_score[l - 1] <= basin_level:
        l -= 1

    r = m
    while r < len(speech_score) - 1 and speech_score[r + 1] <= basin_level:
        r += 1

    if l == r:
        l = max(0, m - 1)
        r = min(len(seg_t) - 1, m + 1)

    new_prev_end = float(seg_t[l])
    new_next_start = float(seg_t[r])

    new_prev_end = max(prev_start, min(new_prev_end, next_end))
    new_next_start = max(prev_start, min(new_next_start, next_end))

    if new_next_start < new_prev_end:
        mid = 0.5 * (new_prev_end + new_next_start)
        new_prev_end = mid
        new_next_start = mid

    return round(new_prev_end, 4), round(new_next_start, 4)

# =========================
# Endpoint refinement config
# =========================
REFINE_FIRST_START = True
REFINE_LAST_END = True

FIRST_START_LEFT_SEC = 0.15
FIRST_START_RIGHT_SEC = 0.45
FIRST_START_MAX_SHIFT_RIGHT_SEC = 0.30
FIRST_START_PREROLL_SEC = 0.02

LAST_END_LEFT_FROM_START_SEC = 0.05
LAST_END_RIGHT_SEC = 0.60
LAST_END_MAX_SHIFT_ABS_SEC = 0.50
LAST_END_POSTROLL_SEC = 0.03

ENDPOINT_SPEECH_LOW_Q = 20
ENDPOINT_SPEECH_HIGH_Q = 75
ENDPOINT_SPEECH_THR_RATIO = 0.35
ENDPOINT_MIN_SPEECH_DUR_SEC = 0.035


def _compute_speech_score_segment(
    log_energy: np.ndarray,
    env: np.ndarray,
    fluct: np.ndarray,
) -> np.ndarray:
    """Process spoken Stroop recordings."""
    z_e = robust_scale(log_energy)
    z_env = robust_scale(env)
    z_fluct = robust_scale(fluct)

    speech_score = (
        WEIGHT_ENERGY * z_e +
        WEIGHT_ENV * z_env +
        WEIGHT_FLUCT * z_fluct
    ).astype(np.float32)

    return moving_average(speech_score, 3)


def _speech_threshold(score: np.ndarray) -> float:
    q_low = float(np.percentile(score, ENDPOINT_SPEECH_LOW_Q))
    q_high = float(np.percentile(score, ENDPOINT_SPEECH_HIGH_Q))
    return q_low + ENDPOINT_SPEECH_THR_RATIO * max(1e-6, q_high - q_low)


def refine_first_token_start(
    times: np.ndarray,
    log_energy: np.ndarray,
    env: np.ndarray,
    fluct: np.ndarray,
    first_item: Dict[str, Any],
) -> Optional[float]:
    """Process spoken Stroop recordings."""
    if first_item.get("start_time") is None or first_item.get("end_time") is None:
        return None

    raw_start = float(first_item["start_time"])
    raw_end = float(first_item["end_time"])

    if raw_end <= raw_start:
        return None

    search_left = max(0.0, raw_start - FIRST_START_LEFT_SEC)
    search_right = min(raw_end, raw_start + FIRST_START_RIGHT_SEC)

    if search_right <= search_left:
        return None

    li = int(np.searchsorted(times, search_left, side="left"))
    ri = int(np.searchsorted(times, search_right, side="right")) - 1

    li = max(0, min(li, len(times) - 1))
    ri = max(0, min(ri, len(times) - 1))

    if ri - li < 6:
        return None

    seg_t = times[li:ri + 1]
    seg_score = _compute_speech_score_segment(
        log_energy[li:ri + 1],
        env[li:ri + 1],
        fluct[li:ri + 1],
    )

    thr = _speech_threshold(seg_score)
    speech_mask = seg_score >= thr
    runs = find_true_runs(speech_mask)

    min_frames = max(2, int(round(ENDPOINT_MIN_SPEECH_DUR_SEC / 0.001)))

    valid_runs = []
    for s, e in runs:
        if (e - s + 1) >= min_frames:
            run_start_t = float(seg_t[s])
            run_end_t = float(seg_t[e])
            if run_end_t >= raw_start:
                valid_runs.append((s, e))

    if not valid_runs:
        return None

    # Processing step.
    s, e = valid_runs[0]
    detected_onset = float(seg_t[s])

    # Processing step.
    new_start = detected_onset - FIRST_START_PREROLL_SEC

    # Processing step.
    lower_bound = raw_start
    upper_bound = min(
        raw_start + FIRST_START_MAX_SHIFT_RIGHT_SEC,
        raw_end - 0.03,
    )

    new_start = max(lower_bound, min(new_start, upper_bound))

    if new_start >= raw_end:
        return None

    return round(new_start, 4)


def refine_last_token_end(
    times: np.ndarray,
    log_energy: np.ndarray,
    env: np.ndarray,
    fluct: np.ndarray,
    last_item: Dict[str, Any],
    audio_duration: float,
) -> Optional[float]:
    """Process spoken Stroop recordings."""
    if last_item.get("start_time") is None or last_item.get("end_time") is None:
        return None

    raw_start = float(last_item["start_time"])
    raw_end = float(last_item["end_time"])

    if raw_end <= raw_start:
        return None

    search_left = max(0.0, raw_start + LAST_END_LEFT_FROM_START_SEC)
    search_right = min(audio_duration, raw_end + LAST_END_RIGHT_SEC)

    if search_right <= search_left:
        return None

    li = int(np.searchsorted(times, search_left, side="left"))
    ri = int(np.searchsorted(times, search_right, side="right")) - 1

    li = max(0, min(li, len(times) - 1))
    ri = max(0, min(ri, len(times) - 1))

    if ri - li < 6:
        return None

    seg_t = times[li:ri + 1]
    seg_score = _compute_speech_score_segment(
        log_energy[li:ri + 1],
        env[li:ri + 1],
        fluct[li:ri + 1],
    )

    thr = _speech_threshold(seg_score)
    speech_mask = seg_score >= thr
    runs = find_true_runs(speech_mask)

    min_frames = max(2, int(round(ENDPOINT_MIN_SPEECH_DUR_SEC / 0.001)))

    valid_runs = []
    for s, e in runs:
        if (e - s + 1) >= min_frames:
            run_start_t = float(seg_t[s])
            run_end_t = float(seg_t[e])

            # Processing step.
            if run_end_t >= raw_start + 0.05:
                valid_runs.append((s, e))

    if not valid_runs:
        return None

    # Processing step.
    s, e = valid_runs[-1]
    detected_offset = float(seg_t[e])

    # Processing step.
    new_end = detected_offset + LAST_END_POSTROLL_SEC

    lower_bound = raw_start + 0.08
    upper_bound = min(audio_duration, raw_end + LAST_END_MAX_SHIFT_ABS_SEC)

    # Processing step.
    new_end = max(lower_bound, min(new_end, upper_bound))

    if abs(new_end - raw_end) > LAST_END_MAX_SHIFT_ABS_SEC:
        return None

    if new_end <= raw_start:
        return None

    return round(new_end, 4)
def refine_boundaries_by_pairwise_features(
    audio_path: str,
    aligned_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Process spoken Stroop recordings."""
    if not aligned_items or len(aligned_items) < 2:
        return aligned_items

    try:
        y, sr = load_audio_mono(audio_path)
        audio_duration = len(y) / float(sr)
        times, log_energy, env, fluct = compute_boundary_features(y, sr)
    except Exception:
        return aligned_items

    refined = [dict(x) for x in aligned_items]

    # Processing step.
    for i in range(len(refined) - 1):
        pair = refine_one_pair_boundary(
            times=times,
            log_energy=log_energy,
            env=env,
            fluct=fluct,
            prev_item=refined[i],
            next_item=refined[i + 1],
        )
        if pair is None:
            continue

        new_end_i, new_start_j = pair
        refined[i]["end_time"] = new_end_i
        refined[i + 1]["start_time"] = new_start_j

        if refined[i]["end_time"] < refined[i]["start_time"]:
            refined[i]["end_time"] = refined[i]["start_time"]

        if refined[i + 1]["start_time"] > refined[i + 1]["end_time"]:
            refined[i + 1]["start_time"] = refined[i + 1]["end_time"]

    # Processing step.
    if REFINE_FIRST_START:
        new_first_start = refine_first_token_start(
            times=times,
            log_energy=log_energy,
            env=env,
            fluct=fluct,
            first_item=refined[0],
        )
        if new_first_start is not None:
            refined[0]["start_time"] = new_first_start

            if refined[0]["start_time"] > refined[0]["end_time"]:
                refined[0]["start_time"] = refined[0]["end_time"]

    # Processing step.
    if REFINE_LAST_END:
        new_last_end = refine_last_token_end(
            times=times,
            log_energy=log_energy,
            env=env,
            fluct=fluct,
            last_item=refined[-1],
            audio_duration=audio_duration,
        )
        if new_last_end is not None:
            refined[-1]["end_time"] = new_last_end

            if refined[-1]["end_time"] < refined[-1]["start_time"]:
                refined[-1]["end_time"] = refined[-1]["start_time"]

    return refined

# =========================
# File processing
# =========================
def make_relative_parts(audio_path: Path, audio_root: str) -> Tuple[str, str, str]:
    rel = audio_path.relative_to(Path(audio_root))
    parts = rel.parts
    subject_folder = parts[0] if len(parts) >= 1 else ""
    session = parts[1] if len(parts) >= 2 else ""
    return subject_folder, session, audio_path.name


def process_one_file(model, audio_path: Path, audio_root: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    subject_folder, session, filename = make_relative_parts(audio_path, audio_root)
    subject_id = subject_id_from_folder(subject_folder)
    task = task_from_filename(filename)

    base_record: Dict[str, Any] = {
        "subject_id": subject_id,
        "subject_folder": subject_folder,
        "session": session,
        "task": task,
        "audio_path": str(audio_path),
        "raw_transcript": "",
        "normalized_text": "",
        "raw_color_candidates": [],
        "cleaned_colors": [],
        "aligned_colors_raw": [],
        "aligned_colors": [],
        "n_raw_color_candidates": 0,
        "n_cleaned_colors": 0,
        "expected_tokens": EXPECTED_TOKENS_PER_TASK,
        "needs_review": False,
        "review_reason": "",
        "error": "",
    }

    token_rows: List[Dict[str, Any]] = []

    try:
        asr_result = call_asr(model, str(audio_path))
        raw_text = asr_result["text"] or ""
        normalized_text = normalize_text(raw_text)
        raw_candidates = extract_color_candidates_from_text(normalized_text)
        cleaned_colors, cleanup_reason = clean_color_sequence(raw_candidates, EXPECTED_TOKENS_PER_TASK)

        aligned_items = filter_aligned_items_to_colors(asr_result.get("time_stamps", []))

        if cleaned_colors and (len(cleaned_colors) > EXPECTED_TOKENS_PER_TASK or not aligned_items or len(aligned_items) < len(cleaned_colors)):
            forced = filter_aligned_items_to_colors(
                align_cleaned_text_if_needed(model, str(audio_path), cleaned_colors)
            )
            if forced:
                aligned_items = forced



        aligned_items_raw = [dict(x) for x in aligned_items]

        if len(aligned_items) >= 2:
            aligned_items = refine_boundaries_by_pairwise_features(
                audio_path=str(audio_path),
                aligned_items=aligned_items,
            )

        review_reasons: List[str] = []
        if cleanup_reason:
            review_reasons.append(cleanup_reason)
        if len(cleaned_colors) != EXPECTED_TOKENS_PER_TASK:
            review_reasons.append(f"cleaned_n={len(cleaned_colors)}")
        if aligned_items and len(aligned_items) != len(cleaned_colors):
            review_reasons.append(f"aligned_n={len(aligned_items)}_cleaned_n={len(cleaned_colors)}")
        if task == "unknown":
            review_reasons.append("unknown_task")

        base_record.update({
            "raw_transcript": raw_text,
            "normalized_text": normalized_text,
            "raw_color_candidates": raw_candidates,
            "cleaned_colors": cleaned_colors,
            "aligned_colors_raw": aligned_items_raw,
            "aligned_colors": aligned_items,
            "n_raw_color_candidates": len(raw_candidates),
            "n_cleaned_colors": len(cleaned_colors),
            "needs_review": len(review_reasons) > 0,
            "review_reason": " | ".join(review_reasons),
        })

        for i, tok in enumerate(cleaned_colors, start=1):
            raw_aligned = aligned_items_raw[i - 1] if i - 1 < len(aligned_items_raw) else {}
            refined_aligned = aligned_items[i - 1] if i - 1 < len(aligned_items) else {}

            start_time_raw = raw_aligned.get("start_time") if isinstance(raw_aligned, dict) else None
            end_time_raw = raw_aligned.get("end_time") if isinstance(raw_aligned, dict) else None
            start_time = refined_aligned.get("start_time") if isinstance(refined_aligned, dict) else None
            end_time = refined_aligned.get("end_time") if isinstance(refined_aligned, dict) else None

            delta_start = None
            if start_time_raw is not None and start_time is not None:
                delta_start = round(float(start_time) - float(start_time_raw), 4)

            delta_end = None
            if end_time_raw is not None and end_time is not None:
                delta_end = round(float(end_time) - float(end_time_raw), 4)

            token_rows.append({
                "subject_id": subject_id,
                "subject_folder": subject_folder,
                "session": session,
                "task": task,
                "audio_path": str(audio_path),
                "token_index": i,
                "token_text": tok,
                "aligned_text_raw": raw_aligned.get("text", "") if isinstance(raw_aligned, dict) else "",
                "aligned_text": refined_aligned.get("text", "") if isinstance(refined_aligned, dict) else "",
                "start_time_raw": start_time_raw,
                "end_time_raw": end_time_raw,
                "start_time": start_time,
                "end_time": end_time,
                "delta_start": delta_start,
                "delta_end": delta_end,
            })

    except Exception as e:
        base_record.update({
            "needs_review": True,
            "review_reason": "exception",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        })

    return base_record, token_rows


# =========================
# Output helpers
# =========================
def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", name)


def save_json_record(out_dir: str, record: Dict[str, Any]) -> None:
    subject_id = record.get("subject_id", "unknown")
    task = record.get("task", "unknown")
    audio_name = sanitize_filename(Path(record.get("audio_path", "unknown.wav")).stem)
    subject_dir = os.path.join(out_dir, "json", subject_id)
    ensure_dir(subject_dir)
    path = os.path.join(subject_dir, f"{audio_name}_{task}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def write_tables(out_dir: str, utterance_records: List[Dict[str, Any]], token_records: List[Dict[str, Any]]) -> None:
    ensure_dir(out_dir)
    utterance_df = pd.DataFrame(utterance_records)
    token_df = pd.DataFrame(token_records)

    utterance_csv = os.path.join(out_dir, "utterance_results.csv")
    token_csv = os.path.join(out_dir, "token_results.csv")
    review_csv = os.path.join(out_dir, "needs_review.csv")

    utterance_df.to_csv(utterance_csv, index=False, encoding="utf-8-sig")
    token_df.to_csv(token_csv, index=False, encoding="utf-8-sig")

    if not utterance_df.empty:
        utterance_df.loc[utterance_df["needs_review"] == True].to_csv(review_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["subject_id", "audio_path", "review_reason", "error"]).to_csv(
            review_csv, index=False, encoding="utf-8-sig"
        )


# =========================
# Main
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="按 qwen3_stroop_pipeline_add.py 的原始步骤进行 Stroop 转写、强制对齐与边界微调。"
    )
    parser.add_argument(
        "--audio-root",
        default=AUDIO_ROOT,
        help="数据集根目录；其下应为 <被试文件夹>/session_1/04_Stroop/*.wav。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="ASR 结果目录。默认建立在数据集根目录的同级目录，命名为 <数据集名>_qwen3_stroop。",
    )
    parser.add_argument("--device", default=DEVICE, help="推理设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--dtype", default=DTYPE, help="推理精度，例如 float32、float16 或 auto。")
    parser.add_argument("--asr-model", default=ASR_MODEL, help="Qwen3-ASR 本地模型目录。")
    parser.add_argument("--aligner-model", default=ALIGNER_MODEL, help="Qwen3 强制对齐器本地模型目录。")
    return parser.parse_args()


def main():
    global AUDIO_ROOT, OUTPUT_DIR, DEVICE, DTYPE, ASR_MODEL, ALIGNER_MODEL
    args = parse_args()
    AUDIO_ROOT = os.path.abspath(args.audio_root)
    OUTPUT_DIR = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        os.path.dirname(AUDIO_ROOT), f"{Path(AUDIO_ROOT).name}_qwen3_stroop"
    )
    DEVICE = args.device
    DTYPE = args.dtype
    ASR_MODEL = args.asr_model
    ALIGNER_MODEL = args.aligner_model

    ensure_dir(OUTPUT_DIR)

    wav_files = find_audio_files(AUDIO_ROOT)
    print(f"共找到 {len(wav_files)} 个 wav 文件。")

    if not ASR_MODEL or not ALIGNER_MODEL:
        raise ValueError("Supply --asr-model and --aligner-model (local model directories)")
    model = load_model(ASR_MODEL, ALIGNER_MODEL, DEVICE, DTYPE)

    utterance_records: List[Dict[str, Any]] = []
    token_records: List[Dict[str, Any]] = []

    for wav_path in tqdm(wav_files, desc="处理中"):
        record, token_rows = process_one_file(model, wav_path, AUDIO_ROOT)
        utterance_records.append(record)
        token_records.extend(token_rows)
        save_json_record(OUTPUT_DIR, record)

    write_tables(OUTPUT_DIR, utterance_records, token_records)

    n_review = sum(1 for x in utterance_records if x.get("needs_review"))
    print("处理完成。")
    print(f"总音频数: {len(utterance_records)}")
    print(f"需要复核: {n_review}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
