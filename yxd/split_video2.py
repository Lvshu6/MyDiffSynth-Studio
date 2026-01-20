import os
import glob
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm


def get_valid_clip_length(total_frames, max_len=33, min_len=5, step=4):
    """
    根据总帧数自动选择合适的片段长度：
    33 → 29 → 25 → 21 → 17 → 13 → 9 → 5
    """
    for L in range(max_len, min_len - 1, -step):
        if total_frames >= L:
            return L
    return min_len  # 最后兜底


def split_video_or_imagefolder_to_5frame_clips(
        input_video_dir: str,
        output_clip_dir: str = "5frame_clips",
        clip_length: int = 33,
        video_format: tuple = ("mp4", "avi", "mov", "mkv", "webm"),
        image_format: tuple = ("jpg", "jpeg", "png", "bmp", "tiff")
):
    """
    将长视频或图片文件夹切分为片段：
    优先使用 clip_length（默认 33），不足则自动尝试 29、25…直到 5。
    """

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

    # -----------------------------
    # 3. 处理视频文件
    # -----------------------------
    for video_idx, video_path in enumerate(video_files, 1):
        video_path = Path(video_path)
        video_name = video_path.stem

        # 跳过已处理
        if any(f.name.startswith(video_name) for f in output_dir.glob("*.mp4")):
            print(f"⏩ 已存在前缀 {video_name} 的片段视频，跳过处理")
            continue

        try:
            frames = iio.imread(video_path)
            total_frames = len(frames)
        except Exception as e:
            print(f"⚠️ 读取视频失败：{e}")
            continue

        # 动态选择片段长度
        valid_len = get_valid_clip_length(total_frames, max_len=clip_length)

        print(f"🎬 视频 {video_name} 总帧数 {total_frames} → 使用片段长度 {valid_len}")

        num_clips = (total_frames + valid_len - 1) // valid_len

        for clip_idx in tqdm(range(num_clips), desc=f"   切割片段"):
            start_idx = clip_idx * valid_len
            end_idx = start_idx + valid_len
            clip_frames = frames[start_idx:end_idx]

            # 不足补齐
            if len(clip_frames) < valid_len:
                pad_frames = np.repeat(clip_frames[-1:], valid_len - len(clip_frames), axis=0)
                clip_frames = np.concatenate([clip_frames, pad_frames], axis=0)

            clip_save_name = f"{video_name}_clip_{clip_idx + 1}.mp4"
            clip_save_path = output_dir / clip_save_name
            iio.imwrite(clip_save_path, clip_frames, fps=5, codec="h264", quality=10)

    # -----------------------------
    # 4. 处理图片文件夹
    # -----------------------------
    for folder_idx, folder in enumerate(image_folders, 1):
        folder_name = folder.name

        if any(f.name.startswith(folder_name) for f in output_dir.glob("*.mp4")):
            print(f"⏩ 已存在前缀 {folder_name} 的片段视频，跳过处理")
            continue

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

        # 动态选择片段长度
        valid_len = get_valid_clip_length(total_frames, max_len=clip_length)

        print(f"🖼️ 图片文件夹 {folder_name} 总帧数 {total_frames} → 使用片段长度 {valid_len}")

        num_clips = (total_frames + valid_len - 1) // valid_len

        for clip_idx in tqdm(range(num_clips), desc=f"   切割片段"):
            start_idx = clip_idx * valid_len
            end_idx = start_idx + valid_len
            clip_frames = frames[start_idx:end_idx]

            if len(clip_frames) < valid_len:
                pad_frames = [clip_frames[-1]] * (valid_len - len(clip_frames))
                clip_frames.extend(pad_frames)

            clip_save_name = f"{folder_name}_clip_{clip_idx + 1}.mp4"
            clip_save_path = output_dir / clip_save_name
            iio.imwrite(clip_save_path, clip_frames, fps=5, codec="h264", quality=10)

    print(f"\n🎉 所有视频和图片文件夹处理完成！片段已保存到：{output_dir}")


if __name__ == "__main__":
    INPUT_VIDEO_DIR = r"/home/suat/yxd/DiffSynth-Studio/data/train1/"
    OUTPUT_CLIP_DIR = r"/home/suat/yxd/DiffSynth-Studio/data/f33"

    split_video_or_imagefolder_to_5frame_clips(
        input_video_dir=INPUT_VIDEO_DIR,
        output_clip_dir=OUTPUT_CLIP_DIR,
        clip_length=33
    )
