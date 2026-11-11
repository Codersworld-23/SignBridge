"""
Data collection and preprocessing utility for SignBridge.
Supports two modes:
1. Batch extraction from downloaded video folders (e.g., INCLUDE video clips).
2. Live interactive webcam recording to record custom ISL training samples.
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLASSES, PROCESSED_DATA_DIR, RAW_DATA_DIR, SEQUENCE_LENGTH
from keypoints_extractor import KeypointsExtractor


def record_live_gestures(
    target_class: str,
    num_samples: int = 15,
    frames_per_sample: int = SEQUENCE_LENGTH,
    camera_index: int = 0
):
    """
    Interactively records ISL gestures using the webcam and saves them as .npy keypoint sequences.
    """
    if target_class not in CLASSES:
        print(f"[ERROR] Target class '{target_class}' not in predefined CLASSES list.")
        return

    save_dir = PROCESSED_DATA_DIR / target_class
    save_dir.mkdir(parents=True, exist_ok=True)

    extractor = KeypointsExtractor()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        return

    existing_files = list(save_dir.glob("sample_*.npy"))
    start_index = len(existing_files)

    print(f"\n=== Recording '{target_class}' Gestures ===")
    print(f"Goal: {num_samples} samples | Frames per sample: {frames_per_sample}")
    print("Press SPACE to start recording each sample, or 'q' to quit.")

    for sample_idx in range(start_index, start_index + num_samples):
        # Countdown / preparation
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            results = extractor.process_frame(frame)
            frame = extractor.draw_landmarks(frame, results)

            # Display instruction prompt
            cv2.putText(
                frame,
                f"Class: {target_class} | Sample: {sample_idx + 1}/{start_index + num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press SPACE when ready to perform gesture",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1
            )
            cv2.imshow("SignBridge Data Recorder", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 32:  # Space
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                extractor.close()
                return

        # Record frames
        sequence = []
        for frame_num in range(frames_per_sample):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            results = extractor.process_frame(frame)
            kps = extractor.extract_keypoints(results)
            sequence.append(kps)

            frame = extractor.draw_landmarks(frame, results)
            cv2.putText(
                frame,
                f"RECORDING... {frame_num + 1}/{frames_per_sample}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            cv2.imshow("SignBridge Data Recorder", frame)
            cv2.waitKey(30)

        # Save sample
        sample_arr = np.array(sequence, dtype=np.float32)
        save_file = save_dir / f"sample_{sample_idx:04d}.npy"
        np.save(save_file, sample_arr)
        print(f"[SAVED] {save_file}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"\n[DONE] Finished recording {num_samples} samples for '{target_class}'.")


def process_video_directory(
    input_video_dir: Path,
    output_processed_dir: Path = PROCESSED_DATA_DIR
):
    """
    Extracts keypoint sequences from all video files (.mp4, .avi) inside a directory
    structured as: input_video_dir / class_name / video.mp4
    """
    extractor = KeypointsExtractor()
    total_processed = 0

    print(f"\n=== Batch Processing Videos from '{input_video_dir}' ===")
    for cls_dir in input_video_dir.iterdir():
        if not cls_dir.is_dir():
            continue

        cls_name = cls_dir.name
        save_dir = output_processed_dir / cls_name
        save_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
        print(f"Processing '{cls_name}': {len(video_files)} videos found")

        for idx, vid_path in enumerate(video_files):
            out_file = save_dir / f"sample_{idx:04d}.npy"
            if out_file.exists():
                continue

            kps_seq = extractor.extract_from_video(str(vid_path), target_frames=SEQUENCE_LENGTH)
            if kps_seq is not None:
                np.save(out_file, kps_seq)
                total_processed += 1

    extractor.close()
    print(f"\n[COMPLETED] Extracted keypoint sequences from {total_processed} videos.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        sign = sys.argv[2] if len(sys.argv) > 2 else CLASSES[0]
        record_live_gestures(target_class=sign, num_samples=10)
    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        source_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW_DATA_DIR
        process_video_directory(source_dir)
    else:
        print("Usage:")
        print("  python collect_data.py record <ClassName>  (Interactive webcam recorder)")
        print("  python collect_data.py batch <VideoFolder>  (Extract from downloaded videos)")
