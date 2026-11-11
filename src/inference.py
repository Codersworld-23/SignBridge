"""
Real-time SignBridge Inference Application.
Integrates:
- Live Webcam Video Capture (OpenCV)
- MediaPipe Holistic Landmark Tracking
- PyTorch Bi-LSTM Neural Sequence Recognition
- Mamdani Fuzzy Inference System (FIS) for Gesture Stabilization
- Dynamic Sentence Construction & HUD Interface
"""

import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

# Import project modules
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLASSES, INPUT_FEATURE_DIM, SEQUENCE_LENGTH,
    MODELS_DIR, RESULTS_DIR, COMMIT_COOLDOWN_FRAMES
)
from keypoints_extractor import KeypointsExtractor
from model import get_model_by_name
from fuzzy_engine import FuzzySignStabilizer


class SignBridgeInference:
    def __init__(self, model_name: str = "bilstm", checkpoint_path: Path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        print(f"[INFO] Initializing SignBridge with architecture '{model_name.upper()}' on {self.device}...")

        # Feature extractor
        self.extractor = KeypointsExtractor()

        # Neural Network (Instantiated via factory method)
        self.model = get_model_by_name(model_name, input_dim=INPUT_FEATURE_DIM, num_classes=len(CLASSES))

        # Auto-resolve checkpoint path if not provided
        if checkpoint_path is None:
            folder_map = {
                "bilstm": RESULTS_DIR / "1_bilstm" / "checkpoint_best.pth",
                "cnngru": RESULTS_DIR / "2_cnngru" / "checkpoint_best.pth",
                "tcn": RESULTS_DIR / "3_tcn" / "checkpoint_best.pth",
                "transformer": RESULTS_DIR / "4_transformer" / "checkpoint_best.pth",
            }
            checkpoint_path = folder_map.get(model_name.lower(), MODELS_DIR / "signbridge_bilstm.pth")

        self.model_loaded = False
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"[INFO] Successfully loaded model checkpoint from: {checkpoint_path}")
        else:
            print(f"[WARNING] No checkpoint found at '{checkpoint_path}'. Running in DEMO / UNTRAINED mode.")
            self.model.to(self.device)
            self.model.eval()

        # Fuzzy Stabilizer
        self.fuzzy = FuzzySignStabilizer()

        # Sliding window sequence buffers
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=15)
        self.sentence = []
        self.last_committed_word = None
        self.cooldown_counter = 0
        self.prev_landmarks = None

    def calculate_motion(self, current_kps: np.ndarray) -> float:
        """
        Calculates normalized inter-frame displacement of hand landmarks.
        Hand keypoints occupy indices 132 to 258 in the 258-dim vector.
        """
        if self.prev_landmarks is None:
            self.prev_landmarks = current_kps
            return 0.0

        # Hand slice (Left hand 63 + Right hand 63)
        hands_curr = current_kps[132:258]
        hands_prev = self.prev_landmarks[132:258]
        self.prev_landmarks = current_kps

        # Check if hands are present
        if np.all(hands_curr == 0) or np.all(hands_prev == 0):
            return 0.0

        # Euclidean distance of coordinates
        displacement = np.linalg.norm(hands_curr - hands_prev)
        # Normalize motion into roughly [0.0, 1.0] range
        motion_index = float(np.clip(displacement / 1.5, 0.0, 1.0))
        return motion_index

    def calculate_stability(self, top_class: str) -> float:
        """
        Calculates the consistency of recent predictions for the candidate class.
        """
        if len(self.recent_predictions) == 0:
            return 0.0
        matches = sum(1 for p in self.recent_predictions if p == top_class)
        return float(matches / len(self.recent_predictions))

    def run_webcam(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {camera_index}")
            return

        print("\n=== SignBridge Live Interface Started ===")
        print("Controls:")
        print("  'c' - Clear current sentence")
        print("  'backspace' - Delete last committed word")
        print("  'q' - Quit application\n")

        fps_time = time.time()
        fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view for natural interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. Landmark Extraction
            results = self.extractor.process_frame(frame)
            kps = self.extractor.extract_keypoints(results)
            self.sequence_buffer.append(kps)

            # Draw visual skeletal landmarks
            frame = self.extractor.draw_landmarks(frame, results)

            # Calculate motion kinematics
            motion = self.calculate_motion(kps)

            pred_label = "Waiting..."
            confidence = 0.0
            stability = 0.0
            fuzzy_result = {
                "decision_score": 0.0,
                "state": "IDLE"
            }

            # 2. Sequence Prediction & Fuzzy Analysis
            if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                input_tensor = torch.tensor(
                    np.array(self.sequence_buffer), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    conf_val, pred_idx = torch.max(probs, dim=-1)
                    confidence = float(conf_val.item())
                    pred_label = CLASSES[pred_idx.item()]

                self.recent_predictions.append(pred_label)
                stability = self.calculate_stability(pred_label)

                # 3. Fuzzy Inference System
                fuzzy_result = self.fuzzy.evaluate(
                    confidence=confidence,
                    stability=stability,
                    motion=motion
                )

                # Cooldown logic for word commitment
                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1

                if (
                    fuzzy_result["state"] == "COMMIT"
                    and self.cooldown_counter == 0
                    and pred_label != "Waiting..."
                ):
                    # Commit word if new or separated by cooldown
                    if len(self.sentence) == 0 or self.sentence[-1] != pred_label:
                        self.sentence.append(pred_label)
                        self.last_committed_word = pred_label
                        self.cooldown_counter = COMMIT_COOLDOWN_FRAMES

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - fps_time + 1e-6)
            fps_time = curr_time

            # 4. HUD Rendering
            frame = self._render_hud(
                frame, pred_label, confidence, stability, motion, fuzzy_result, fps
            )

            cv2.imshow("SignBridge - Real-Time Neuro-Fuzzy ISL Translation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence.clear()
            elif key == 8:  # Backspace
                if len(self.sentence) > 0:
                    self.sentence.pop()

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _render_hud(
        self,
        frame: np.ndarray,
        pred_label: str,
        confidence: float,
        stability: float,
        motion: float,
        fuzzy_result: dict,
        fps: float
    ) -> np.ndarray:
        h, w, _ = frame.shape

        # Top Sentence Banner
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        sentence_str = " ".join(self.sentence) if self.sentence else "[No signs committed yet]"
        cv2.putText(
            frame,
            f"Output: {sentence_str}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255) if self.sentence else (160, 160, 160),
            2,
            cv2.LINE_AA
        )

        # Left Diagnostics Card (Fuzzy & Neural Metrics)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 75), (340, 290), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Card Border
        cv2.rectangle(frame, (15, 75), (340, 290), (70, 70, 70), 1)

        cv2.putText(frame, "SignBridge Telemetry", (25, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Candidate Sign & Confidence
        cv2.putText(frame, f"Sign: {pred_label}", (25, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Neural Conf: {confidence * 100:.1f}%", (25, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

        # Fuzzy Inputs
        cv2.putText(frame, f"Temporal Stability: {stability * 100:.1f}%", (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(frame, f"Hand Motion: {motion:.2f}", (25, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        # Fuzzy Output State
        state = fuzzy_result.get("state", "IDLE")
        score = fuzzy_result.get("decision_score", 0.0)

        if state == "COMMIT":
            state_color = (0, 255, 0)
        elif state == "HOLD":
            state_color = (0, 215, 255)  # Amber/Yellow
        else:
            state_color = (128, 128, 128)  # Gray

        cv2.putText(frame, f"Fuzzy Score: {score:.2f}", (25, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"State: [{state}]", (25, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

        # Bottom Controls & FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, "Press 'c': Clear | 'q': Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge Real-Time Webcam Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        choices=["bilstm", "cnngru", "tcn", "transformer"],
        help="Model architecture to run with webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    app = SignBridgeInference(model_name=args.model)
    app.run_webcam(camera_index=args.camera)
