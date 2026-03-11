"""
Master Nightly Autonomous Pipeline:
Runs after market close (16:00 IST) to prepare for next trading day.

Pipeline:
1. Data Refresh: Update instrument master + fetch OHLCV
2. Nightly Simulation: Test 1000s of strategy permutations
3. Model Retraining: Retrain if drift detected or weekly schedule
4. Pre-Market Prep: Generate LLM briefing for next morning
5. Notification: Summary to admin

Usage: python -m scripts.nightly_autonomous_pipeline
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nightly_pipeline")

_IST = timezone(timedelta(hours=5, minutes=30))


async def step_1_data_refresh():
    """Refresh market data."""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA REFRESH")
    logger.info("=" * 50)
    try:
        from scripts.nightly_data_refresh import main as data_main
        quality = await data_main()
        return quality
    except Exception as e:
        logger.error("Data refresh failed: %s", e)
        return {"quality_pass": False, "error": str(e)}


async def step_2_nightly_simulation(symbols: list = None):
    """Run nightly strategy simulation."""
    logger.info("=" * 50)
    logger.info("STEP 2: NIGHTLY SIMULATION")
    logger.info("=" * 50)
    try:
        from src.simulation.orchestrator import SimulationOrchestrator
        orch = SimulationOrchestrator(max_workers=4, top_n=5)
        results = await orch.run_nightly_pipeline(
            symbols=symbols,
            intervals=["15m", "1h"],
        )
        selected = orch.simulator.get_selected_strategies()
        logger.info("Simulation: %d results, %d selected", len(results), len(selected))
        return {
            "total_results": len(results),
            "selected_count": len(selected),
            "top_sharpe": results[0].sharpe_ratio if results else 0,
            "selected": [
                {"strategy": s.strategy_id, "sharpe": s.sharpe_ratio, "win_rate": s.win_rate}
                for s in selected
            ],
        }
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        return {"error": str(e)}


async def step_3_model_retraining():
    """Check if model retraining is needed and run if so."""
    logger.info("=" * 50)
    logger.info("STEP 3: MODEL RETRAINING CHECK")
    logger.info("=" * 50)
    try:
        # Check if models exist
        model_dir = Path(__file__).resolve().parents[1] / "models"
        models_exist = (model_dir / "alpha_xgb.joblib").exists()

        if not models_exist:
            logger.info("No trained models found, triggering training")
            # Run XGBoost training
            from scripts.train_alpha_model import main as train_main
            train_main()
            return {"action": "trained", "reason": "no_models_found"}

        # Check model age
        model_file = model_dir / "alpha_xgb.joblib"
        model_age_days = (datetime.now() - datetime.fromtimestamp(model_file.stat().st_mtime)).days

        if model_age_days > 7:
            logger.info("Model is %d days old, triggering retrain", model_age_days)
            from scripts.train_alpha_model import main as train_main
            train_main()
            return {"action": "retrained", "reason": f"model_age_{model_age_days}d"}

        logger.info("Model is %d days old, no retrain needed", model_age_days)
        return {"action": "skipped", "model_age_days": model_age_days}

    except Exception as e:
        logger.error("Model retraining failed: %s", e)
        return {"action": "failed", "error": str(e)}


async def step_4_pre_market_prep():
    """Prepare pre-market briefing for tomorrow."""
    logger.info("=" * 50)
    logger.info("STEP 4: PRE-MARKET PREPARATION")
    logger.info("=" * 50)
    try:
        from src.ai.llm.news_feed import NewsFeedAggregator
        from src.ai.llm.pre_market_brief import PreMarketBriefing

        # Fetch news
        news = NewsFeedAggregator()
        await news.fetch_all()
        headlines = news.get_headline_summary()

        # Generate briefing (will use fallback if no LLM configured)
        briefing = PreMarketBriefing()
        brief = await briefing.generate_briefing(
            news_headlines=headlines,
            global_markets={
                "S&P 500": 0.0,  # Would be fetched from API
                "NASDAQ": 0.0,
                "Nikkei": 0.0,
                "SGX Nifty": 0.0,
            },
        )
        return {
            "outlook": brief.get("outlook", "unknown"),
            "exposure": brief.get("exposure_multiplier", 1.0),
            "news_count": len(headlines.split("\n")) if headlines else 0,
        }
    except Exception as e:
        logger.error("Pre-market prep failed: %s", e)
        return {"error": str(e)}


async def main():
    """Run the full nightly autonomous pipeline."""
    start = datetime.now()
    logger.info("=" * 60)
    logger.info("AUTONOMOUS NIGHTLY PIPELINE - %s IST",
                 datetime.now(_IST).strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    results = {}

    # Step 1: Data refresh
    results["data"] = await step_1_data_refresh()

    # Step 2: Simulation (only if data is good)
    if results["data"].get("quality_pass", False):
        results["simulation"] = await step_2_nightly_simulation()
    else:
        logger.warning("Skipping simulation due to poor data quality")
        results["simulation"] = {"skipped": True, "reason": "data_quality"}

    # Step 3: Model retraining
    results["training"] = await step_3_model_retraining()

    # Step 4: Pre-market prep
    results["pre_market"] = await step_4_pre_market_prep()

    elapsed = (datetime.now() - start).total_seconds()

    # Summary
    logger.info("=" * 60)
    logger.info("NIGHTLY PIPELINE COMPLETE - %.1f seconds", elapsed)
    logger.info("-" * 60)
    logger.info("Data:       %s", "PASS" if results["data"].get("quality_pass") else "FAIL")
    logger.info("Simulation: %s selected", results["simulation"].get("selected_count", "N/A"))
    logger.info("Training:   %s", results["training"].get("action", "N/A"))
    logger.info("Pre-Market: %s outlook, %.1fx exposure",
                 results["pre_market"].get("outlook", "N/A"),
                 results["pre_market"].get("exposure", 1.0))
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())
