"""Statistical performance decay detection."""
import numpy as np


class DecayDetector:
    """
    Detect strategy decay: e.g. recent Sharpe/returns significantly worse than
    historical. Uses simple split: first half vs second half of lookback.
    """

    def __init__(self, lookback: int = 20, min_periods: int = 10):
        self.lookback = lookback
        self.min_periods = min_periods

    def detect(self, returns: np.ndarray) -> bool:
        """
        Return True if decay detected (recent performance worse than earlier).
        """
        if len(returns) < self.min_periods:
            return False
        r = returns[-self.lookback:] if len(returns) >= self.lookback else returns
        n = len(r)
        mid = n // 2
        first = r[:mid]
        second = r[mid:]
        if len(first) < 2 or len(second) < 2:
            return False
        mean_first = np.mean(first)
        mean_second = np.mean(second)
        std_first = np.std(first)
        std_second = np.std(second)
        # When both halves have zero std, decay = recent mean lower than prior mean
        if std_first < 1e-12 and std_second < 1e-12:
            return mean_second < mean_first - 1e-9
        if std_first < 1e-12:
            sharpe_first = 0.0
        else:
            sharpe_first = mean_first / (std_first + 1e-12)
        if std_second < 1e-12:
            sharpe_second = 0.0
        else:
            sharpe_second = mean_second / (std_second + 1e-12)
        # Decay if recent Sharpe < 50% of prior
        return sharpe_second < 0.5 * sharpe_first and sharpe_first > 0
