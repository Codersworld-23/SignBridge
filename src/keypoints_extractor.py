"""
MediaPipe keypoint extractor for SignBridge.
Extracts normalized 3D landmarks for pose and hands (left and right),
producing a flattened feature vector of size 258 per frame.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional


class KeypointsExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Initializes MediaPipe Holistic model.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            refine_face_landmarks=False
        )

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Processes an RGB image and returns MediaPipe results.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

    def extract_keypoints(self, results) -> np.ndarray:
        """
        Extracts concatenated landmark vector from detection results:
        - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
        - Left hand: 21 landmarks * 3 (x, y, z) = 63
        - Right hand: 21 landmarks * 3 (x, y, z) = 63
        Total dimension = 258
        """
        if results.pose_landmarks:
            pose = np.array(
                [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
            ).flatten()
        else:
            pose = np.zeros(33 * 4, dtype=np.float32)

        if results.left_hand_landmarks:
            lh = np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        if results.right_hand_landmarks:
            rh = np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        return np.concatenate([pose, lh, rh]).astype(np.float32)

    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Renders landmark skeleton on the image with distinct colors.
        """
        # Draw Pose connections
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw Left Hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

        # Draw Right Hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        return image

    def extract_from_video(
        self,
        video_path: str,
        target_frames: int = 30
    ) -> Optional[np.ndarray]:
        """
        Reads a video file, extracts landmarks for each frame, and normalizes
        the sequence length to `target_frames` via uniform temporal sampling.
        Returns array of shape (target_frames, 258).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        raw_frames_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            kps = self.extract_keypoints(results)
            raw_frames_keypoints.append(kps)

        cap.release()

        total_frames = len(raw_frames_keypoints)
        if total_frames == 0:
            return None

        # Resample sequence to target_frames
        if total_frames == target_frames:
            sampled_sequence = np.array(raw_frames_keypoints)
        elif total_frames > target_frames:
            # Uniform downsampling
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            sampled_sequence = np.array([raw_frames_keypoints[i] for i in indices])
        else:
            # Linear interpolation / pad with last frame
            pad_count = target_frames - total_frames
            last_frame = raw_frames_keypoints[-1]
            padded = raw_frames_keypoints + [last_frame] * pad_count
            sampled_sequence = np.array(padded)

        return sampled_sequence.astype(np.float32)

    def close(self):
        self.holistic.close()
