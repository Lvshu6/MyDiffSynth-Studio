import os
import glob
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------
# 1. 动态切分长度（逐级减法）
# ---------------------------------------------------------
def dynamic_split_lengths(total_frames, lengths=[33, 29, 25, 21, 17, 13, 9, 5]):
    """
    根据剩余帧数，依次尝试使用最大可行的片段长度。
    例如：55 → [33, 21, 5]
    """
    result = []
    remain = total_frames

    while remain > 0:
        for L in lengths:
            if remain >= L:
                result.append(L)
                remain -= L
                break
        else:
            # remain < 最小长度 → 用最小长度补齐
            result.append(lengths[-1])
            remain = 0

    return result


# ---------------------------------------------------------
# 2. 向前回溯补齐（视频帧）
# ---------------------------------------------------------
def backward_pad_frames(full_frames, clip_frames, target_len):
    """
    从原序列中向前回溯补齐，保持顺序。
    full_frames: 原始完整帧序列 (numpy)
    clip_frames: 当前片段 (numpy)
    """
    cur = len(clip_frames)
    if cur >= target_len:
        return clip_frames

    need = target_len - cur

    # 找到 clip 的起始位置（逐帧比较）
    start_idx = None
    for i in range(len(full_frames)):
        if np.array_equal(full_frames[i], clip_frames[0]):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("clip_frames[0] not found in full_frames")

    pad_start = max(0, start_idx - need)
    pad = full_frames[pad_start:start_idx]

    return np.concatenate([pad, clip_frames], axis=0)


# ---------------------------------------------------------
# 3. 向前回溯补齐（图片帧列表）
# ---------------------------------------------------------
def backward_pad_list(full_list, clip_list, target_len):
    cur = len(clip_list)
    if cur >= target_len:
        return clip_list

    need = target_len - cur

    # 找到 clip 的起始位置（逐帧比较）
    start_idx = None
    for i, f in enumerate(full_list):
        if np.array_equal(f, clip_list[0]):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("clip_list[0] not found in full_list")

    pad_start = max(0, start_idx - need)
    pad = full_list[pad_start:start_idx]

    return pad + clip_list


# ---------------------------------------------------------
# 4. 主切分函数
# ---------------------------------------------------------
def split_video_or_imagefolder_dynamic(
        input_video_dir: str,
        output_clip_dir: str = "clips_dynamic",
        lengths=(33, 29, 25, 21, 17, 13, 9, 5),
        video_format=("mp4", "avi", "mov", "mkv", "webm"),
        image_format=("jpg", "jpeg", "png", "bmp", "tiff")
):
    input_dir = Path(input_video_dir)
    output_dir = Path(output_clip_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 输入目录：{input_dir}")
    print(f"📁 输出片段目录：{output_dir}")

    # 1. 查找视频文件
    video_files = []
    for fmt in video_format:
        video_files.extend(glob.glob(str(input_dir / f"*.{fmt}")))

    # 2. 查找图片文件夹
    image_folders = [f for f in input_dir.iterdir() if f.is_dir()]

    if not video_files and not image_folders:
        raise FileNotFoundError(f"在 {input_dir} 中未找到视频文件或图片文件夹")

    # ---------------------------------------------------------
    # 5. 处理视频文件
    # ---------------------------------------------------------
    for video_path in video_files:
        video_path = Path(video_path)
        video_name = video_path.stem

        try:
            frames = iio.imread(video_path)
            total_frames = len(frames)
        except Exception as e:
            print(f"⚠️ 读取视频失败：{e}")
            continue

        clip_lengths = dynamic_split_lengths(total_frames, lengths)
        print(f"🎬 视频 {video_name} 总帧数 {total_frames} → 切分方案 {clip_lengths}")

        start = 0
        for idx, L in enumerate(tqdm(clip_lengths, desc=f"   切割片段")):
            end = start + L
            clip_frames = frames[start:end]

            # 向前回溯补齐
            clip_frames = backward_pad_frames(frames, clip_frames, L)

            save_name = f"{video_name}_clip_{idx + 1}.mp4"
            save_path = output_dir / save_name
            iio.imwrite(save_path, clip_frames, fps=5, codec="h264", quality=10)

            start = end

    # ---------------------------------------------------------
    # 6. 处理图片文件夹
    # ---------------------------------------------------------
    for folder in image_folders:
        folder_name = folder.name

        # 找到所有图片
        frame_paths = []
        for ext in image_format:
            frame_paths.extend(glob.glob(str(folder / f"*.{ext}")))

        if not frame_paths:
            print(f"⚠️ 文件夹 {folder_name} 没有图片，跳过")
            continue

        # 按数字排序
        def sort_key(path):
            digits = ''.join(filter(str.isdigit, os.path.basename(path)))
            return int(digits) if digits else path

        frame_paths.sort(key=sort_key)
        frames = [iio.imread(fp) for fp in frame_paths]
        total_frames = len(frames)

        clip_lengths = dynamic_split_lengths(total_frames, lengths)
        print(f"🖼️ 图片文件夹 {folder_name} 总帧数 {total_frames} → 切分方案 {clip_lengths}")

        start = 0
        for idx, L in enumerate(tqdm(clip_lengths, desc=f"   切割片段")):
            end = start + L
            clip_frames = frames[start:end]

            # 向前回溯补齐
            clip_frames = backward_pad_list(frames, clip_frames, L)

            save_name = f"{folder_name}_clip_{idx + 1}.mp4"
            save_path = output_dir / save_name
            iio.imwrite(save_path, clip_frames, fps=5, codec="h264", quality=10)

            start = end

    print(f"\n🎉 所有视频和图片文件夹处理完成！片段已保存到：{output_dir}")


# ---------------------------------------------------------
# 7. 主入口
# ---------------------------------------------------------
if __name__ == "__main__":
    INPUT_VIDEO_DIR = r"/home/suat/yxd/DiffSynth-Studio/data/orgtest/"
    OUTPUT_CLIP_DIR = r"/home/suat/yxd/DiffSynth-Studio/data/new2"

    split_video_or_imagefolder_dynamic(
        input_video_dir=INPUT_VIDEO_DIR,
        output_clip_dir=OUTPUT_CLIP_DIR,
        lengths=[33, 29, 25, 21, 17, 13, 9, 5]
    )
