import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import opensmile


# =========================================================
# 配置
# =========================================================
AUDIO_ROOT = str(Path(__file__).resolve().parents[2] / "data" / "example" / "audio")
TOKEN_RESULTS_CSV = str(Path(__file__).resolve().parents[2] / "data" / "example" / "token_results.csv")

# 最终输出：尽量与原始报告一致
FINAL_OUTPUT_CSV_FILE_PATH = str(Path(__file__).resolve().parents[2] / "results" / "recomputed" / "example_features.csv")

# 使用 token_results.csv 中哪一套时间戳
# 微调后时间戳
AUTO_START_COL = "start_time"
AUTO_END_COL = "end_time"

# 如果想改成微调前时间戳，改成下面两行：
# AUTO_START_COL = "start_time_raw"
# AUTO_END_COL = "end_time_raw"

# 只保留你关心的 3 个任务
PROCESS_TASKS = ["Hanzi", "STT", "Dots"]
TASK_FILENAME_MAP = {"Hanzi": "hanzi", "STT": "STT", "Dots": "dots"}
TASK_FROM_CSV_MAP = {"hanzi": "Hanzi", "dots": "Dots", "stt": "STT", "STT": "STT", "W": "Hanzi", "C": "Dots", "CW": "STT"}

VALID_KEYWORDS = {"红", "黄", "蓝", "绿"}

# OpenSMILE 配置：与原始脚本一致
OPENSMILE_FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
OPENSMILE_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals

# token_results.csv 中可能的列名候选
SUBJECT_COLS = ["subject_id", "Subject_ID", "subject"]
TASK_COLS = ["task", "Task"]
ITEM_COLS = ["token_index", "Item_Index", "trial_index", "idx"]
TOKEN_TEXT_COLS = ["token_text", "word", "label", "aligned_text"]

audio_file_cache: Dict[tuple, Optional[str]] = {}

# 与 2.2缺失值处理.py 保持一致：这些基础列不参与 OpenSMILE 特征插补。



def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_word(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    mapping = {"紅": "红", "黃": "黄", "藍": "蓝", "綠": "绿"}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def normalize_task_for_report(x: str) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[s]
    sl = s.lower()
    if sl in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[sl]
    return None


def find_audio_file(subject_id: str, task_report_name: str) -> Optional[str]:
    """
    在 AUDIO_ROOT 下查找原始音频文件。
    文件命名形如：
    EXAMPLE_ID_stroop_dots.wav
    EXAMPLE_ID_stroop_STT.wav
    EXAMPLE_ID_stroop_hanzi.wav
    """
    key = (subject_id, task_report_name)
    if key in audio_file_cache:
        return audio_file_cache[key]

    filename_part = TASK_FILENAME_MAP.get(task_report_name, "")
    target_name = f"{subject_id}_stroop_{filename_part}.wav"

    # 优先按固定层级找
    candidates = list(Path(AUDIO_ROOT).glob(f"*_{subject_id}/session_1/04_Stroop/{target_name}"))
    if candidates:
        audio_file_cache[key] = str(candidates[0])
        return str(candidates[0])

    # 兜底全盘搜索
    aliases = {"Hanzi": "W", "Dots": "C", "STT": "CW"}
    alias_name = f"{subject_id}_stroop_{aliases[task_report_name]}.wav"
    for root, _, files in os.walk(AUDIO_ROOT):
        if target_name in files or alias_name in files:
            p = os.path.join(root, target_name if target_name in files else alias_name)
            audio_file_cache[key] = p
            return p

    audio_file_cache[key] = None
    return None


def load_audio(audio_path: str):
    signal, sr = sf.read(audio_path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal, sr


def slice_signal(signal: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(0, int(round(start_s * sr)))
    end_idx = min(len(signal), int(round(end_s * sr)))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    seg = signal[start_idx:end_idx]
    return seg.astype(np.float32, copy=False)


def prepare_timestamp_df() -> pd.DataFrame:
    print("\n--- 阶段 1/3: 从 token_results.csv 获取时间戳并计算 Duration, Gap_Before ---")

    if not os.path.exists(TOKEN_RESULTS_CSV):
        raise FileNotFoundError(f"未找到 token_results.csv: {TOKEN_RESULTS_CSV}")

    df = pd.read_csv(TOKEN_RESULTS_CSV)

    subject_col = find_first_existing_col(df, SUBJECT_COLS)
    task_col = find_first_existing_col(df, TASK_COLS)
    item_col = find_first_existing_col(df, ITEM_COLS)
    word_col = find_first_existing_col(df, TOKEN_TEXT_COLS)

    missing = []
    if subject_col is None:
        missing.append("subject_id")
    if task_col is None:
        missing.append("task")
    if item_col is None:
        missing.append("token_index")
    if word_col is None:
        missing.append("token_text")
    if AUTO_START_COL not in df.columns:
        missing.append(AUTO_START_COL)
    if AUTO_END_COL not in df.columns:
        missing.append(AUTO_END_COL)
    if missing:
        raise ValueError(f"token_results.csv 缺少必要列: {missing}")

    work = df.copy()
    work["Subject_ID"] = work[subject_col].astype(str)
    work["Task"] = work[task_col].map(normalize_task_for_report)
    work["Item_Index"] = pd.to_numeric(work[item_col], errors="coerce")
    work["Auto_Word"] = work[word_col].map(normalize_word)
    work["Absolute_Word_Start"] = pd.to_numeric(work[AUTO_START_COL], errors="coerce")
    work["Absolute_Word_End"] = pd.to_numeric(work[AUTO_END_COL], errors="coerce")

    # 过滤合法任务、合法颜色词、合法时间戳
    work = work[work["Task"].isin(PROCESS_TASKS)].copy()
    work = work[work["Auto_Word"].isin(VALID_KEYWORDS)].copy()
    work = work[work["Absolute_Word_Start"].notna() & work["Absolute_Word_End"].notna()].copy()
    work = work[work["Absolute_Word_End"] > work["Absolute_Word_Start"]].copy()
    work = work[work["Item_Index"].notna()].copy()

    work["Item_Index"] = work["Item_Index"].astype(int)

    # 只保留每个被试-任务的前12个 token，并按 Item_Index 排序
    work = work.sort_values(["Subject_ID", "Task", "Item_Index"]).copy()
    work = work.groupby(["Subject_ID", "Task"], as_index=False, group_keys=False).head(12).copy()

    # 重新计算每个任务内的 1..12，以尽量贴近原报告结构
    work["Item_Index"] = work.groupby(["Subject_ID", "Task"]).cumcount() + 1

    # Duration / Gap_Before
    work["Duration"] = work["Absolute_Word_End"] - work["Absolute_Word_Start"]
    work["Gap_Before"] = 0.0

    for (sid, task), idx in work.groupby(["Subject_ID", "Task"]).groups.items():
        sub = work.loc[idx].sort_values("Item_Index")
        prev_end = None
        gaps = []
        for _, row in sub.iterrows():
            if prev_end is None:
                gaps.append(float(row["Absolute_Word_Start"]))
            else:
                gaps.append(float(row["Absolute_Word_Start"] - prev_end))
            prev_end = float(row["Absolute_Word_End"])
        work.loc[sub.index, "Gap_Before"] = gaps

    out_cols = [
        "Subject_ID", "Task", "Item_Index",
        "Absolute_Word_Start", "Absolute_Word_End",
        "Duration", "Gap_Before", "Auto_Word"
    ]
    work = work[out_cols].copy()

    print(f"自动时间戳收集完成，共 {len(work)} 条记录。")
    if not work.empty:
        print("任务分布:")
        for task, count in work.groupby("Task").size().items():
            print(f"  {task}: {count} 条记录")

    return work


def extract_opensmile_features_in_memory(df_timestamps: pd.DataFrame) -> pd.DataFrame:
    print("\n--- 阶段 2/3: 直接根据 token_results.csv 时间戳在内存中切片并提取OpenSMILE特征 (不分割、不保存音频) ---")
    print("使用OpenSMILE特征集: eGeMAPSv02 (Functionals)")

    # 测试一条，确定特征列
    smile_test = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
    )

    feature_columns = []
    test_done = False

    for _, row in df_timestamps.iterrows():
        sid = str(row["Subject_ID"])
        task = row["Task"]
        audio_path = find_audio_file(sid, task)
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            signal, sr = load_audio(audio_path)
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                continue
            test_features = smile_test.process_signal(seg, sr)
            if not test_features.empty:
                feature_columns = test_features.columns.tolist()
                print(f"测试音频 {os.path.basename(audio_path)} 特征提取成功:")
                print(f"  特征形状: {test_features.shape}")
                print(f"  特征数量: {len(feature_columns)}")
                print(f"  前10个特征: {feature_columns[:10]}")
                print(f"  后10个特征: {feature_columns[-10:]}")
                test_done = True
                break
        except Exception as e:
            print(f"测试片段特征提取失败: {e}")

    if not test_done or not feature_columns:
        print("警告：无法提取测试片段特征，将返回空特征表。")
        return pd.DataFrame(columns=["Subject_ID", "Task", "Item_Index"])

    smile = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
        num_workers=1,
    )

    all_features_data = []

    # 以音频为单位缓存，避免反复读盘
    loaded_audio_cache: Dict[str, tuple] = {}

    for _, row in tqdm(df_timestamps.iterrows(), total=len(df_timestamps), desc="提取OpenSMILE特征"):
        sid = str(row["Subject_ID"])
        task = row["Task"]
        item_idx = int(row["Item_Index"])
        start_s = float(row["Absolute_Word_Start"])
        end_s = float(row["Absolute_Word_End"])

        audio_path = find_audio_file(sid, task)

        feature_row = {
            "Subject_ID": sid,
            "Task": task,
            "Item_Index": item_idx,
        }

        if not audio_path or not os.path.exists(audio_path):
            print(f"警告：未找到原始音频 Subject={sid}, Task={task}")
            for col in feature_columns:
                feature_row[col] = np.nan
            all_features_data.append(feature_row)
            continue

        try:
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(audio_path)

            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, start_s, end_s)

            if len(seg) == 0:
                print(f"警告：切片为空 Subject={sid}, Task={task}, Item={item_idx}")
                for col in feature_columns:
                    feature_row[col] = np.nan
                all_features_data.append(feature_row)
                continue

            features_df_single = smile.process_signal(seg, sr)

            if not features_df_single.empty:
                features = features_df_single.iloc[0].values
                for col_name, feature_val in zip(feature_columns, features):
                    feature_row[col_name] = feature_val
            else:
                print(f"警告：特征提取结果为空 Subject={sid}, Task={task}, Item={item_idx}")
                for col in feature_columns:
                    feature_row[col] = np.nan

        except Exception as e:
            print(f"错误：处理 Subject={sid}, Task={task}, Item={item_idx} 失败: {e}")
            for col in feature_columns:
                feature_row[col] = np.nan

        all_features_data.append(feature_row)

    features_df = pd.DataFrame(all_features_data)

    expected_columns = ["Subject_ID", "Task", "Item_Index"] + feature_columns
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = np.nan
    features_df = features_df[expected_columns]

    return features_df


def merge_and_save(df_timestamps: pd.DataFrame, features_df: pd.DataFrame):
    print("\n--- 阶段 3/3: 合并所有数据并保存最终CSV ---")

    df_timestamps = df_timestamps.copy()
    features_df = features_df.copy()

    df_timestamps["Subject_ID"] = df_timestamps["Subject_ID"].astype(str)
    features_df["Subject_ID"] = features_df["Subject_ID"].astype(str)

    final_df = pd.merge(
        df_timestamps,
        features_df,
        on=["Subject_ID", "Task", "Item_Index"],
        how="left"
    )

    # 为了尽量与原报告一致，把 Auto_Word 放到后面；保留但不干扰原核心结构
    ordered_front = [
        "Subject_ID", "Task", "Item_Index",
        "Absolute_Word_Start", "Absolute_Word_End",
        "Duration", "Gap_Before"
    ]
    other_cols = [c for c in final_df.columns if c not in ordered_front]
    final_df = final_df[ordered_front + other_cols]

    os.makedirs(os.path.dirname(FINAL_OUTPUT_CSV_FILE_PATH), exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_CSV_FILE_PATH, index=False, encoding="utf-8-sig")

    opensmile_cols = [col for col in features_df.columns if col not in ["Subject_ID", "Task", "Item_Index"] and col in final_df.columns]
    print(f"\n最终CSV已保存至: {FINAL_OUTPUT_CSV_FILE_PATH}")
    print("\n=== 数据摘要 ===")
    print(f"总记录数: {len(final_df)}")
    print(f"总列数: {len(final_df.columns)}")
    print(f"  基本特征: {len(final_df.columns) - len(opensmile_cols)} 列")
    print(f"  OpenSMILE特征: {len(opensmile_cols)} 列")

    if opensmile_cols:
        missing_values = final_df[opensmile_cols].isna().sum().sum()
        total_cells = len(final_df) * len(opensmile_cols)
        missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        print(f"OpenSMILE特征缺失值: {missing_values} / {total_cells} ({missing_percentage:.1f}%)")

    basic_cols = [col for col in final_df.columns if col not in opensmile_cols]
    print(f"\n基本特征列: {basic_cols}")

    return final_df


def process_stroop_data() -> None:
    """Extract un-imputed item-level features from the provided token times."""
    timestamps = prepare_timestamp_df()
    if timestamps.empty:
        raise ValueError("No valid timestamp rows")
    features = extract_opensmile_features_in_memory(timestamps)
    if features.empty:
        raise ValueError("No acoustic features were extracted")
    merge_and_save(timestamps, features)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract un-imputed item-level eGeMAPS features from token timestamps."
    )
    parser.add_argument(
        "--audio-root",
        default=AUDIO_ROOT,
        help="数据集根目录；其下应为 <被试文件夹>/session_1/04_Stroop/*.wav。",
    )
    parser.add_argument(
        "--asr-output-dir",
        default=None,
        help="R.1 产生 token_results.csv 的目录；默认使用 <数据集名>_qwen3_stroop。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="特征 CSV 保存目录；默认与 ASR 输出目录相同。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    AUDIO_ROOT = os.path.abspath(args.audio_root)
    default_asr_dir = str(Path(__file__).resolve().parents[2] / "data" / "example")
    asr_output_dir = os.path.abspath(args.asr_output_dir or default_asr_dir)
    output_dir = os.path.abspath(args.output_dir or str(Path(__file__).resolve().parents[2] / "results" / "recomputed"))
    TOKEN_RESULTS_CSV = os.path.join(asr_output_dir, "token_results.csv")
    FINAL_OUTPUT_CSV_FILE_PATH = os.path.join(output_dir, "Stroop_features_add.csv")

    try:
        import opensmile
        print(f"opensmile-python 版本: {opensmile.__version__}")
        print("OpenSMILE 特征集: eGeMAPSv02")
        print("OpenSMILE 特征级别: Functionals")
    except ImportError:
        print("错误：未安装 opensmile-python 库。请先 pip install opensmile")
        raise

    process_stroop_data()
