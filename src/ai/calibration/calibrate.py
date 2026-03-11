"""
Platt scaling: P_cal = sigmoid(β0 + β1 * logit(p)).
Isotonic regression: non-parametric monotonic mapping from score to probability.
Reliability curve: binned predicted prob vs realized frequency (for monitoring).
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling: fit sigmoid(β0 + β1 * logit(p)) on validation (y, p)."""

    def __init__(self):
        self.b0: float = 0.0
        self.b1: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> None:
        from scipy import optimize

        logit_p = logit(p)

        def nll(params):
            b0, b1 = params
            q = sigmoid(b0 + b1 * logit_p)
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = optimize.minimize(nll, [0.0, 1.0], method="L-BFGS-B")
        self.b0, self.b1 = res.x

    def transform(self, p: np.ndarray) -> np.ndarray:
        return sigmoid(self.b0 + self.b1 * logit(np.asarray(p)))


class IsotonicCalibrator:
    """Isotonic regression: monotonic mapping. Fit on (p, y); predict with interpolate."""

    def __init__(self):
        self._mapping: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> None:
        try:
            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p, y)
            self._mapping = (np.linspace(0, 1, 101), ir.predict(np.linspace(0, 1, 101)))
        except ImportError:
            self._mapping = (np.linspace(0, 1, 11), np.linspace(0, 1, 11))

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._mapping is None:
            return np.asarray(p)
        x_ref, y_ref = self._mapping
        return np.interp(np.asarray(p), x_ref, y_ref)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Binned mean predicted prob and mean realized frequency.
    Returns (bin_edges_mid, mean_predicted_prob, mean_realized_freq).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bins[:-1] + bins[1:]) / 2
    mean_pred = np.zeros(n_bins)
    mean_real = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if np.any(mask):
            mean_pred[i] = np.mean(y_prob[mask])
            mean_real[i] = np.mean(y_true[mask])
        else:
            mean_pred[i] = bin_mids[i]
            mean_real[i] = np.nan
    return bin_mids, mean_pred, mean_real
