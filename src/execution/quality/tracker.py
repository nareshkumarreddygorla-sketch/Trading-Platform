"""
Execution Quality Tracker (Enhanced).

Comprehensive execution quality measurement:
  - Slippage measurement: fill price vs arrival price
  - Implementation shortfall calculation
  - TWAP/VWAP benchmark comparison
  - Per-algo execution quality scoring
  - Daily execution quality report generation

Rolling window metrics for position sizing feedback and strategy disable decisions.
"""

import collections
import logging
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Aggregate execution quality metrics."""

    slippage_ratio: float  # realized / expected (e.g. > 1.5 -> bad)
    rejection_rate: float
    partial_fill_rate: float
    mean_latency_ms: float
    n_fills: int
    n_rejections: int
    n_orders: int


@dataclass
class SlippageMeasurement:
    """Single fill slippage measurement."""

    symbol: str
    side: str  # BUY or SELL
    arrival_price: float  # Price when decision was made (signal price)
    fill_price: float  # Actual execution price
    quantity: int
    slippage_bps: float  # Signed: positive = adverse, negative = favorable
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    algo: str = "direct"  # Algorithm used: direct, twap, vwap, iceberg
    order_id: str = ""


@dataclass
class ImplementationShortfallResult:
    """Implementation shortfall breakdown for a single order."""

    symbol: str
    side: str
    decision_price: float  # Price at signal/decision time
    arrival_price: float  # Price when order entered the market
    fill_price: float  # Actual fill price
    quantity: int
    # Components (all in bps)
    delay_cost_bps: float  # (arrival - decision) / decision * 10000
    execution_cost_bps: float  # (fill - arrival) / arrival * 10000
    total_shortfall_bps: float  # (fill - decision) / decision * 10000
    # Dollar values
    paper_return: float  # What we'd have if filled at decision price
    actual_return: float  # What we actually got
    shortfall_value: float  # paper_return - actual_return
    algo: str = "direct"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BenchmarkComparison:
    """Comparison of execution vs TWAP/VWAP benchmark."""

    exec_id: str
    symbol: str
    side: str
    algo: str
    benchmark_type: str  # "TWAP" or "VWAP"
    benchmark_price: float  # Theoretical benchmark price
    achieved_price: float  # Actual weighted average fill price
    deviation_bps: float  # Signed: positive = worse than benchmark
    total_quantity: int
    filled_quantity: int
    fill_rate: float  # filled / total
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AlgoQualityScore:
    """Per-algorithm execution quality score."""

    algo: str
    n_executions: int
    mean_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    mean_shortfall_bps: float
    fill_rate: float  # Average fill rate (1.0 = all filled)
    benchmark_deviation_bps: float
    score: float  # 0-100 composite score (higher = better)


@dataclass
class DailyExecutionReport:
    """Daily execution quality summary."""

    date: str  # YYYY-MM-DD
    total_orders: int
    total_fills: int
    total_rejections: int
    rejection_rate: float
    partial_fill_rate: float
    mean_slippage_bps: float
    median_slippage_bps: float
    mean_shortfall_bps: float
    mean_latency_ms: float
    algo_scores: dict[str, AlgoQualityScore] = field(default_factory=dict)
    worst_slippage: SlippageMeasurement | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ExecutionQualityTracker:
    """
    Enhanced execution quality tracking with:
    - Rolling window slippage, rejection, and partial fill metrics
    - Implementation shortfall calculation
    - TWAP/VWAP benchmark comparison
    - Per-algo scoring
    - Daily report generation

    Thread-safe via deque operations (GIL-protected for append/popleft).
    """

    def __init__(self, window: int = 100):
        self.window = window

        # --- Original rolling metrics ---
        self._expected_slip: collections.deque[float] = collections.deque(maxlen=window)
        self._realized_slip: collections.deque[float] = collections.deque(maxlen=window)
        self._partial_fill: collections.deque[bool] = collections.deque(maxlen=window)
        self._order_outcomes: collections.deque[int] = collections.deque(maxlen=window)
        self._latency_ms: collections.deque[float] = collections.deque(maxlen=window)
        self._slippage_ratio_threshold: float = 1.5
        self._rejection_rate_threshold: float = 0.2
        self._partial_fill_rate_threshold: float = 0.3
        self._size_multiplier_on_degrade: float = 0.7

        # --- Enhanced tracking ---
        # Slippage measurements (bounded)
        self._slippage_records: collections.deque[SlippageMeasurement] = collections.deque(maxlen=window * 5)

        # Implementation shortfall records
        self._shortfall_records: collections.deque[ImplementationShortfallResult] = collections.deque(maxlen=window * 5)

        # Benchmark comparisons
        self._benchmark_records: collections.deque[BenchmarkComparison] = collections.deque(maxlen=window * 2)

        # Per-algo tracking
        self._algo_slippage: dict[str, collections.deque[float]] = {}
        self._algo_fill_rates: dict[str, collections.deque[float]] = {}
        self._algo_shortfalls: dict[str, collections.deque[float]] = {}
        self._algo_counts: dict[str, int] = {}

    # --- Original Methods (preserved) ---

    def record_fill(
        self,
        expected_slippage_bps: float,
        realized_slippage_bps: float,
        partial_fill: bool = False,
        latency_ms: float | None = None,
    ) -> None:
        self._order_outcomes.append(0)  # 0 = non-rejection
        self._expected_slip.append(expected_slippage_bps)
        self._realized_slip.append(realized_slippage_bps)
        self._partial_fill.append(partial_fill)
        if latency_ms is not None:
            self._latency_ms.append(latency_ms)

    def record_rejection(self) -> None:
        self._order_outcomes.append(1)  # 1 = rejection

    def get_slippage_ratio(self) -> float:
        if not self._expected_slip or sum(self._expected_slip) <= 0:
            return 1.0
        return sum(self._realized_slip) / sum(self._expected_slip)

    def get_rejection_rate(self) -> float:
        if not self._order_outcomes:
            return 0.0
        return sum(self._order_outcomes) / len(self._order_outcomes)

    def get_partial_fill_rate(self) -> float:
        if not self._partial_fill:
            return 0.0
        return sum(self._partial_fill) / len(self._partial_fill)

    def get_metrics(self) -> QualityMetrics:
        sr = self.get_slippage_ratio()
        rr = self.get_rejection_rate()
        pfr = self.get_partial_fill_rate()
        lat = sum(self._latency_ms) / len(self._latency_ms) if self._latency_ms else 0.0
        return QualityMetrics(
            slippage_ratio=sr,
            rejection_rate=rr,
            partial_fill_rate=pfr,
            mean_latency_ms=lat,
            n_fills=len(self._expected_slip),
            n_rejections=sum(self._order_outcomes),
            n_orders=len(self._order_outcomes),
        )

    def recommend_size_multiplier(self) -> float:
        """Return multiplier for position size (e.g. 0.7 if execution degraded)."""
        sr = self.get_slippage_ratio()
        rr = self.get_rejection_rate()
        if sr >= self._slippage_ratio_threshold or rr >= self._rejection_rate_threshold:
            return self._size_multiplier_on_degrade
        return 1.0

    def recommend_disable(self) -> bool:
        """True if rejection or partial fill rate over threshold (strategy/symbol disable)."""
        if self.get_rejection_rate() >= self._rejection_rate_threshold:
            return True
        if self.get_partial_fill_rate() >= self._partial_fill_rate_threshold:
            return True
        return False

    # --- Enhanced: Slippage Measurement ---

    def record_slippage(
        self,
        symbol: str,
        side: str,
        arrival_price: float,
        fill_price: float,
        quantity: int,
        algo: str = "direct",
        order_id: str = "",
    ) -> SlippageMeasurement:
        """
        Record a fill with arrival price for slippage measurement.

        Slippage is signed: positive = adverse (paid more / received less than arrival).
        """
        if arrival_price <= 0:
            slippage_bps = 0.0
        elif side == "BUY":
            slippage_bps = (fill_price - arrival_price) / arrival_price * 10000
        else:  # SELL
            slippage_bps = (arrival_price - fill_price) / arrival_price * 10000

        measurement = SlippageMeasurement(
            symbol=symbol,
            side=side,
            arrival_price=arrival_price,
            fill_price=fill_price,
            quantity=quantity,
            slippage_bps=slippage_bps,
            algo=algo,
            order_id=order_id,
        )
        self._slippage_records.append(measurement)

        # Update per-algo tracking
        self._ensure_algo_deques(algo)
        self._algo_slippage[algo].append(slippage_bps)

        return measurement

    # --- Enhanced: Implementation Shortfall ---

    def calculate_implementation_shortfall(
        self,
        symbol: str,
        side: str,
        decision_price: float,
        arrival_price: float,
        fill_price: float,
        quantity: int,
        algo: str = "direct",
    ) -> ImplementationShortfallResult:
        """
        Calculate implementation shortfall decomposition.

        Implementation Shortfall = Delay Cost + Execution Cost
          - Delay Cost: cost of waiting between decision and market entry
          - Execution Cost: cost of actual execution vs arrival price

        All costs in basis points, signed (positive = adverse).
        """
        if decision_price <= 0:
            decision_price = arrival_price  # fallback

        # Signed based on side
        if side == "BUY":
            delay_cost_bps = (arrival_price - decision_price) / decision_price * 10000
            execution_cost_bps = (fill_price - arrival_price) / arrival_price * 10000
            total_shortfall_bps = (fill_price - decision_price) / decision_price * 10000
            paper_return = quantity * decision_price
            actual_return = quantity * fill_price
            shortfall_value = actual_return - paper_return  # positive = paid more
        else:  # SELL
            delay_cost_bps = (decision_price - arrival_price) / decision_price * 10000
            execution_cost_bps = (arrival_price - fill_price) / arrival_price * 10000
            total_shortfall_bps = (decision_price - fill_price) / decision_price * 10000
            paper_return = quantity * decision_price
            actual_return = quantity * fill_price
            shortfall_value = paper_return - actual_return  # positive = received less

        result = ImplementationShortfallResult(
            symbol=symbol,
            side=side,
            decision_price=decision_price,
            arrival_price=arrival_price,
            fill_price=fill_price,
            quantity=quantity,
            delay_cost_bps=round(delay_cost_bps, 2),
            execution_cost_bps=round(execution_cost_bps, 2),
            total_shortfall_bps=round(total_shortfall_bps, 2),
            paper_return=round(paper_return, 2),
            actual_return=round(actual_return, 2),
            shortfall_value=round(shortfall_value, 2),
            algo=algo,
        )
        self._shortfall_records.append(result)

        # Update per-algo tracking
        self._ensure_algo_deques(algo)
        self._algo_shortfalls[algo].append(total_shortfall_bps)

        return result

    # --- Enhanced: TWAP/VWAP Benchmark Comparison ---

    def record_benchmark_comparison(
        self,
        exec_id: str,
        symbol: str,
        side: str,
        algo: str,
        benchmark_type: str,
        benchmark_price: float,
        achieved_price: float,
        total_quantity: int,
        filled_quantity: int,
    ) -> BenchmarkComparison:
        """
        Record comparison of execution vs TWAP/VWAP benchmark.

        Deviation is signed: positive = worse than benchmark.
        """
        if benchmark_price <= 0:
            deviation_bps = 0.0
        elif side == "BUY":
            deviation_bps = (achieved_price - benchmark_price) / benchmark_price * 10000
        else:
            deviation_bps = (benchmark_price - achieved_price) / benchmark_price * 10000

        fill_rate = filled_quantity / total_quantity if total_quantity > 0 else 0.0

        comparison = BenchmarkComparison(
            exec_id=exec_id,
            symbol=symbol,
            side=side,
            algo=algo,
            benchmark_type=benchmark_type,
            benchmark_price=round(benchmark_price, 4),
            achieved_price=round(achieved_price, 4),
            deviation_bps=round(deviation_bps, 2),
            total_quantity=total_quantity,
            filled_quantity=filled_quantity,
            fill_rate=round(fill_rate, 4),
        )
        self._benchmark_records.append(comparison)

        # Update per-algo fill rate
        self._ensure_algo_deques(algo)
        self._algo_fill_rates[algo].append(fill_rate)

        return comparison

    # --- Enhanced: Per-Algo Scoring ---

    def _ensure_algo_deques(self, algo: str) -> None:
        """Lazily initialize per-algo deques."""
        if algo not in self._algo_slippage:
            max_len = self.window * 2
            self._algo_slippage[algo] = collections.deque(maxlen=max_len)
            self._algo_fill_rates[algo] = collections.deque(maxlen=max_len)
            self._algo_shortfalls[algo] = collections.deque(maxlen=max_len)
            self._algo_counts[algo] = 0
        self._algo_counts[algo] = self._algo_counts.get(algo, 0) + 1

    def get_algo_quality_score(self, algo: str) -> AlgoQualityScore | None:
        """
        Calculate quality score for a specific algorithm.

        Score (0-100):
          - 40 pts: slippage (lower = better)
          - 30 pts: fill rate (higher = better)
          - 30 pts: shortfall (lower = better)
        """
        slippages = list(self._algo_slippage.get(algo, []))
        fill_rates = list(self._algo_fill_rates.get(algo, []))
        shortfalls = list(self._algo_shortfalls.get(algo, []))

        if not slippages:
            return None

        mean_slip = statistics.mean(slippages) if slippages else 0.0
        median_slip = statistics.median(slippages) if slippages else 0.0
        sorted_slip = sorted(slippages)
        p95_idx = int(len(sorted_slip) * 0.95)
        p95_slip = sorted_slip[min(p95_idx, len(sorted_slip) - 1)] if sorted_slip else 0.0

        mean_fill = statistics.mean(fill_rates) if fill_rates else 1.0
        mean_shortfall = statistics.mean(shortfalls) if shortfalls else 0.0

        # Benchmark deviation from benchmark records
        algo_benchmarks = [b.deviation_bps for b in self._benchmark_records if b.algo == algo]
        bench_dev = statistics.mean(algo_benchmarks) if algo_benchmarks else 0.0

        # Composite score (0-100)
        # Slippage component (40 pts): 0 bps = 40, >50 bps = 0
        slip_score = max(0, 40 - (abs(mean_slip) / 50) * 40)
        # Fill rate component (30 pts): 100% = 30, 0% = 0
        fill_score = mean_fill * 30
        # Shortfall component (30 pts): 0 bps = 30, >100 bps = 0
        short_score = max(0, 30 - (abs(mean_shortfall) / 100) * 30)
        composite = round(slip_score + fill_score + short_score, 1)

        return AlgoQualityScore(
            algo=algo,
            n_executions=self._algo_counts.get(algo, len(slippages)),
            mean_slippage_bps=round(mean_slip, 2),
            median_slippage_bps=round(median_slip, 2),
            p95_slippage_bps=round(p95_slip, 2),
            mean_shortfall_bps=round(mean_shortfall, 2),
            fill_rate=round(mean_fill, 4),
            benchmark_deviation_bps=round(bench_dev, 2),
            score=composite,
        )

    # --- Enhanced: Daily Report ---

    def generate_daily_report(self, date_str: str | None = None) -> DailyExecutionReport:
        """
        Generate daily execution quality report.

        Args:
            date_str: Date string (YYYY-MM-DD). Defaults to today UTC.
        """
        if date_str is None:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        metrics = self.get_metrics()

        # Slippage stats from enhanced records
        all_slippages = [r.slippage_bps for r in self._slippage_records]
        mean_slip = statistics.mean(all_slippages) if all_slippages else 0.0
        median_slip = statistics.median(all_slippages) if all_slippages else 0.0

        # Shortfall stats
        all_shortfalls = [r.total_shortfall_bps for r in self._shortfall_records]
        mean_shortfall = statistics.mean(all_shortfalls) if all_shortfalls else 0.0

        # Per-algo scores
        algo_scores = {}
        for algo in self._algo_slippage:
            score = self.get_algo_quality_score(algo)
            if score:
                algo_scores[algo] = score

        # Worst slippage
        worst = None
        if self._slippage_records:
            worst = max(self._slippage_records, key=lambda r: r.slippage_bps)

        report = DailyExecutionReport(
            date=date_str,
            total_orders=metrics.n_orders,
            total_fills=metrics.n_fills,
            total_rejections=metrics.n_rejections,
            rejection_rate=metrics.rejection_rate,
            partial_fill_rate=metrics.partial_fill_rate,
            mean_slippage_bps=round(mean_slip, 2),
            median_slippage_bps=round(median_slip, 2),
            mean_shortfall_bps=round(mean_shortfall, 2),
            mean_latency_ms=round(metrics.mean_latency_ms, 2),
            algo_scores=algo_scores,
            worst_slippage=worst,
        )

        logger.info(
            "Daily execution report [%s]: orders=%d fills=%d rejections=%d "
            "mean_slip=%.1fbps mean_shortfall=%.1fbps algos=%s",
            date_str,
            report.total_orders,
            report.total_fills,
            report.total_rejections,
            mean_slip,
            mean_shortfall,
            list(algo_scores.keys()),
        )

        return report

    # --- Utility ---

    def get_recent_slippages(self, n: int = 20) -> list[SlippageMeasurement]:
        """Return the N most recent slippage measurements."""
        records = list(self._slippage_records)
        return records[-n:]

    def get_recent_shortfalls(self, n: int = 20) -> list[ImplementationShortfallResult]:
        """Return the N most recent implementation shortfall records."""
        records = list(self._shortfall_records)
        return records[-n:]

    def get_benchmark_comparisons(self, n: int = 20) -> list[BenchmarkComparison]:
        """Return the N most recent benchmark comparisons."""
        records = list(self._benchmark_records)
        return records[-n:]
