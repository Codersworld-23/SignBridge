"""
SignBridge Real-Time Webcam Translation Interface.
Integrates:
- Live video stream acquisition via OpenCV
- Real-time 3D landmark extraction via Google MediaPipe Holistic (258 features)
- Neural sequence classification (Bi-LSTM, CNN-BiGRU, Res-TCN, or Transformer)
- 3-Input Mamdani Fuzzy Inference System (FIS) for gesture stabilization and flicker rejection
- Dynamic sentence construction with cooldown debouncing

Usage:
    python webcam_live.py                     # Runs default Bi-LSTM or available checkpoint
    python webcam_live.py --model transformer  # Runs Transformer model
    python webcam_live.py --camera 1          # Use external USB camera
"""

import sys
import argparse
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from config import CLASSES, INPUT_FEATURE_DIM, SEQUENCE_LENGTH, RESULTS_DIR, COMMIT_COOLDOWN_FRAMES
from keypoints_extractor import KeypointsExtractor
from model import get_model_by_name
from fuzzy_engine import FuzzySignStabilizer


class SignBridgeLiveApp:
    def __init__(self, model_name: str = "bilstm", camera_index: int = 0):
        self.camera_index = camera_index
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing SignBridge with architecture '{model_name.upper()}' on {self.device}...")

        # Initialize MediaPipe Holistic Extractor
        self.extractor = KeypointsExtractor()

        # Instantiate Selected Architecture
        self.model = get_model_by_name(model_name, input_dim=INPUT_FEATURE_DIM, num_classes=len(CLASSES))

        # Checkpoint resolution priority
        checkpoint_candidates = [
            RESULTS_DIR / f"1_{model_name}" / "checkpoint_best.pth",
            RESULTS_DIR / f"2_{model_name}" / "checkpoint_best.pth",
            RESULTS_DIR / f"3_{model_name}" / "checkpoint_best.pth",
            RESULTS_DIR / f"4_{model_name}" / "checkpoint_best.pth",
            PROJECT_ROOT / "models" / f"signbridge_{model_name}.pth",
            PROJECT_ROOT / "models" / "signbridge_bilstm.pth",
        ]

        self.model_loaded = False
        for cp in checkpoint_candidates:
            if cp.exists():
                try:
                    ckpt = torch.load(cp, map_location=self.device)
                    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                    self.model.load_state_dict(state)
                    self.model.to(self.device)
                    self.model.eval()
                    self.model_loaded = True
                    print(f"[SUCCESS] Loaded trained weights from: {cp}")
                    break
                except Exception as e:
                    print(f"[NOTE] Checkpoint {cp.name} structure mismatch: {e}")

        if not self.model_loaded:
            print("\n" + "=" * 65)
            print("[NOTICE] Model checkpoint not found or still training.")
            print("The application will run in SKELETAL + FUZZY TELEMETRY PREVIEW MODE.")
            print(f"To train this model first, run: python train_single.py --model {model_name}")
            print("=" * 65 + "\n")
            self.model.to(self.device)
            self.model.eval()

        # Mamdani Fuzzy Inference Engine
        self.fuzzy = FuzzySignStabilizer()

        # Temporal Sliding Buffers
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=15)
        self.sentence = []
        self.cooldown = 0
        self.prev_kps = None

    def _calc_motion(self, current_kps: np.ndarray) -> float:
        """Computes Euclidean coordinate velocity of hand landmarks."""
        if self.prev_kps is None:
            self.prev_kps = current_kps
            return 0.0

        hands_curr = current_kps[132:258]
        hands_prev = self.prev_kps[132:258]
        self.prev_kps = current_kps

        if np.all(hands_curr == 0) or np.all(hands_prev == 0):
            return 0.0

        displacement = np.linalg.norm(hands_curr - hands_prev)
        return float(np.clip(displacement / 1.5, 0.0, 1.0))

    def _calc_stability(self, candidate_label: str) -> float:
        """Ratio of recent frames predicting the candidate label."""
        if not self.prediction_history:
            return 0.0
        return float(sum(1 for p in self.prediction_history if p == candidate_label) / len(self.prediction_history))

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Unable to access camera device at index {self.camera_index}")
            return

        print("\n=== SignBridge Live Translation Running ===")
        print("Controls:")
        print("  'c'         : Clear sentence buffer")
        print("  'backspace' : Delete last committed sign")
        print("  'q'         : Exit application\n")

        fps_time = time.time()
        fps = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. Extract 3D skeletal landmarks
            results = self.extractor.process_frame(frame)
            kps = self.extractor.extract_keypoints(results)
            self.sequence_buffer.append(kps)
            frame = self.extractor.draw_landmarks(frame, results)

            # Compute kinematic motion
            motion_idx = self._calc_motion(kps)

            pred_label = "Buffering..." if len(self.sequence_buffer) < SEQUENCE_LENGTH else "Detecting..."
            confidence = 0.0
            stability = 0.0
            fuzzy_res = {"decision_score": 0.0, "state": "IDLE"}

            # 2. Sequential Neural Inference + Fuzzy Stabilization
            if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                x_tensor = torch.tensor(
                    np.array(self.sequence_buffer), dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(x_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    conf_val, pred_idx = torch.max(probs, dim=-1)
                    confidence = float(conf_val.item())
                    pred_label = CLASSES[pred_idx.item()]

                self.prediction_history.append(pred_label)
                stability = self._calc_stability(pred_label)

                # Fuzzy Evaluation
                fuzzy_res = self.fuzzy.evaluate(confidence=confidence, stability=stability, motion=motion_idx)

                # Debounce cooldown
                if self.cooldown > 0:
                    self.cooldown -= 1

                if (
                    fuzzy_res["state"] == "COMMIT"
                    and self.cooldown == 0
                    and self.model_loaded
                ):
                    if len(self.sentence) == 0 or self.sentence[-1] != pred_label:
                        self.sentence.append(pred_label)
                        self.cooldown = COMMIT_COOLDOWN_FRAMES

            # Compute FPS
            now = time.time()
            fps = 1.0 / (now - fps_time + 1e-6)
            fps_time = now

            # 3. Render Diagnostics HUD
            frame = self._render_hud(frame, pred_label, confidence, stability, motion_idx, fuzzy_res, fps)

            cv2.imshow(f"SignBridge Live Translation [{self.model_name.upper()}]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.sentence.clear()
            elif key == 8:  # Backspace
                if self.sentence:
                    self.sentence.pop()

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _render_hud(self, frame, pred_label, conf, stab, motion, fuzzy_res, fps):
        h, w, _ = frame.shape

        # Top Sentence Ribbon
        cv2.rectangle(frame, (0, 0), (w, 55), (25, 25, 25), -1)
        sentence_text = " ".join(self.sentence) if self.sentence else "[No signs committed yet]"
        cv2.putText(frame, f"Translation: {sentence_text}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255) if self.sentence else (150, 150, 150), 2)

        # Left Telemetry Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 70), (350, 310), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (15, 70), (350, 310), (60, 60, 60), 1)

        cv2.putText(frame, f"SignBridge HUD ({self.model_name.upper()})", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
        cv2.putText(frame, f"Sign: {pred_label}", (25, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Neural Conf: {conf * 100:.1f}%", (25, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 220, 255), 1)
        cv2.putText(frame, f"Temporal Stability: {stab * 100:.1f}%", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(frame, f"Hand Velocity: {motion:.2f}", (25, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        state = fuzzy_res.get("state", "IDLE")
        score = fuzzy_res.get("decision_score", 0.0)
        state_color = (0, 255, 0) if state == "COMMIT" else ((0, 215, 255) if state == "HOLD" else (130, 130, 130))

        cv2.putText(frame, f"Fuzzy Score: {score:.2f}", (25, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Fuzzy State: [{state}]", (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.78, state_color, 2)

        # Footer
        status_note = "Model: ACTIVE" if self.model_loaded else "Model: UNTRAINED PREVIEW"
        cv2.putText(frame, status_note, (w - 240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.model_loaded else (0, 165, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, "Press 'c': Clear | 'q': Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge Real-Time Webcam Translation")
    parser.add_argument(
        "--model",
        type=str,
        default="bilstm",
        choices=["bilstm", "cnngru", "tcn", "transformer"],
        help="Architecture to load for inference"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    app = SignBridgeLiveApp(model_name=args.model, camera_index=args.camera)
    app.run()
