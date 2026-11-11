"""
Fuzzy Inference System (FIS) for SignBridge.
Integrates soft computing / fuzzy logic into the sign recognition pipeline to handle:
1. Prediction uncertainty (neural network softmax confidence)
2. Temporal consistency (frame stability in a sliding window)
3. Hand movement kinematics (motion index between consecutive frames)

Outputs a defuzzified Decision Score and linguistic state:
- COMMIT (Accept gesture into sentence)
- HOLD (Transitional gesture / buffering)
- REJECT (Noise / idle state)
"""

import numpy as np
from typing import Dict, Tuple, Any


def trapmf(x: np.ndarray, abcd: list) -> np.ndarray:
    """Trapezoidal membership function."""
    a, b, c, d = abcd
    y = np.zeros_like(x, dtype=float)
    # Left slope
    if b > a:
        idx = np.logical_and(a < x, x < b)
        y[idx] = (x[idx] - a) / (b - a)
    # Plateau
    idx = np.logical_and(b <= x, x <= c)
    y[idx] = 1.0
    # Right slope
    if d > c:
        idx = np.logical_and(c < x, x < d)
        y[idx] = (d - x[idx]) / (d - c)
    return y


def trimf(x: np.ndarray, abc: list) -> np.ndarray:
    """Triangular membership function."""
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    if b > a:
        idx = np.logical_and(a < x, x <= b)
        y[idx] = (x[idx] - a) / (b - a)
    if c > b:
        idx = np.logical_and(b < x, x < c)
        y[idx] = (c - x[idx]) / (c - b)
    y[x == b] = 1.0
    return y


class FuzzySignStabilizer:
    """
    Mamdani-style Fuzzy Inference System for Sign Language Recognition.
    Calculates dynamic confidence and suppresses flickering / transitional gestures.
    """

    def __init__(self):
        # Discretized universes of discourse
        self.u_conf = np.linspace(0.0, 1.0, 101)
        self.u_stab = np.linspace(0.0, 1.0, 101)
        self.u_motion = np.linspace(0.0, 1.0, 101)
        self.u_decision = np.linspace(0.0, 1.0, 201)

        # 1. Input: Neural Network Confidence (P_conf)
        self.mf_conf = {
            "low": trapmf(self.u_conf, [0.0, 0.0, 0.35, 0.55]),
            "medium": trimf(self.u_conf, [0.40, 0.65, 0.85]),
            "high": trapmf(self.u_conf, [0.70, 0.85, 1.0, 1.0])
        }

        # 2. Input: Temporal Frame Stability (S_stab)
        self.mf_stab = {
            "flickering": trapmf(self.u_stab, [0.0, 0.0, 0.25, 0.45]),
            "moderate": trimf(self.u_stab, [0.35, 0.60, 0.80]),
            "stable": trapmf(self.u_stab, [0.65, 0.80, 1.0, 1.0])
        }

        # 3. Input: Hand Motion / Velocity (V_motion)
        self.mf_motion = {
            "still": trapmf(self.u_motion, [0.0, 0.0, 0.15, 0.35]),
            "moving": trimf(self.u_motion, [0.20, 0.45, 0.70]),
            "rapid": trapmf(self.u_motion, [0.55, 0.75, 1.0, 1.0])
        }

        # Output: Decision Score (D_score)
        self.mf_decision = {
            "reject": trapmf(self.u_decision, [0.0, 0.0, 0.20, 0.40]),
            "hold": trimf(self.u_decision, [0.30, 0.50, 0.70]),
            "accept": trapmf(self.u_decision, [0.60, 0.75, 1.0, 1.0])
        }

    def _membership_grade(self, universe: np.ndarray, mf_dict: Dict[str, np.ndarray], val: float) -> Dict[str, float]:
        """Interpolates membership grade for a single crisp scalar input."""
        val = np.clip(val, universe[0], universe[-1])
        grades = {}
        for term, curve in mf_dict.items():
            grades[term] = float(np.interp(val, universe, curve))
        return grades

    def evaluate(self, confidence: float, stability: float, motion: float) -> Dict[str, Any]:
        """
        Executes Mamdani inference and centroid defuzzification.
        
        Args:
            confidence: Max Softmax probability [0.0, 1.0]
            stability: Proportion of recent frames agreeing on top label [0.0, 1.0]
            motion: Inter-frame landmark coordinate displacement [0.0, 1.0]
            
        Returns:
            Dictionary with decision score, linguistic state, rule activations, and crisp values.
        """
        # 1. Fuzzification
        mu_c = self._membership_grade(self.u_conf, self.mf_conf, confidence)
        mu_s = self._membership_grade(self.u_stab, self.mf_stab, stability)
        mu_m = self._membership_grade(self.u_motion, self.mf_motion, motion)

        # 2. Rule Evaluation (Mamdani Min operator for AND)
        # Consequent: Accept
        r_accept_1 = min(mu_c["high"], mu_s["stable"], mu_m["still"])
        r_accept_2 = min(mu_c["high"], mu_s["moderate"], mu_m["still"])
        r_accept_3 = min(mu_c["high"], mu_s["stable"], mu_m["moving"])
        alpha_accept = max(r_accept_1, r_accept_2, r_accept_3)

        # Consequent: Hold / Transition
        r_hold_1 = min(mu_c["medium"], mu_s["stable"], mu_m["still"])
        r_hold_2 = min(mu_c["medium"], mu_s["moderate"], mu_m["still"])
        r_hold_3 = min(mu_c["high"], mu_m["rapid"])
        r_hold_4 = min(mu_c["medium"], mu_s["moderate"], mu_m["moving"])
        alpha_hold = max(r_hold_1, r_hold_2, r_hold_3, r_hold_4)

        # Consequent: Reject / Noise
        r_reject_1 = mu_c["low"]
        r_reject_2 = mu_s["flickering"]
        r_reject_3 = min(mu_m["rapid"], mu_c["medium"])
        r_reject_4 = min(mu_c["low"], mu_m["still"])
        alpha_reject = max(r_reject_1, r_reject_2, r_reject_3, r_reject_4)

        # 3. Aggregation (Max union of clipped outputs)
        out_accept = np.minimum(alpha_accept, self.mf_decision["accept"])
        out_hold = np.minimum(alpha_hold, self.mf_decision["hold"])
        out_reject = np.minimum(alpha_reject, self.mf_decision["reject"])

        aggregated = np.maximum(out_reject, np.maximum(out_hold, out_accept))

        # 4. Defuzzification (Centroid / Center of Gravity)
        sum_weights = np.sum(aggregated)
        if sum_weights == 0:
            decision_score = 0.0
        else:
            decision_score = float(np.sum(self.u_decision * aggregated) / sum_weights)

        # Classify linguistic decision
        if decision_score >= 0.65:
            state = "COMMIT"
        elif decision_score >= 0.35:
            state = "HOLD"
        else:
            state = "REJECT"

        return {
            "decision_score": round(decision_score, 3),
            "state": state,
            "crisp_inputs": {
                "confidence": round(confidence, 3),
                "stability": round(stability, 3),
                "motion": round(motion, 3)
            },
            "rule_strengths": {
                "alpha_accept": round(alpha_accept, 3),
                "alpha_hold": round(alpha_hold, 3),
                "alpha_reject": round(alpha_reject, 3)
            }
        }
