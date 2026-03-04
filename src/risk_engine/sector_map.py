"""
NSE sector classification for concentration monitoring.

Maps all NSE stocks to GICS-equivalent sectors using:
  - NSE industry classification from bhavcopy
  - Hardcoded top-200 fallback for reliable coverage
  - Daily refresh from NSE data
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# NSE industry → normalised sector mapping
INDUSTRY_TO_SECTOR: Dict[str, str] = {
    # Banking & Finance
    "FINANCIAL SERVICES": "Banking & Finance",
    "BANKS": "Banking & Finance",
    "PRIVATE SECTOR BANK": "Banking & Finance",
    "PUBLIC SECTOR BANK": "Banking & Finance",
    "BANK - PRIVATE": "Banking & Finance",
    "BANK - PUBLIC": "Banking & Finance",
    "FINANCE": "Banking & Finance",
    "HOUSING FINANCE": "Banking & Finance",
    "NBFC": "Banking & Finance",
    "INSURANCE": "Banking & Finance",
    "STOCK/COMMODITY BROKERS": "Banking & Finance",
    "FINANCIAL INSTITUTION": "Banking & Finance",

    # IT
    "IT - SOFTWARE": "Information Technology",
    "IT - SERVICES": "Information Technology",
    "IT CONSULTING & SOFTWARE": "Information Technology",
    "IT": "Information Technology",
    "SOFTWARE": "Information Technology",
    "INFORMATION TECHNOLOGY": "Information Technology",
    "COMPUTERS - SOFTWARE": "Information Technology",
    "COMPUTERS - HARDWARE": "Information Technology",

    # Pharma & Healthcare
    "PHARMACEUTICALS": "Pharma & Healthcare",
    "HEALTHCARE": "Pharma & Healthcare",
    "HEALTHCARE SERVICES": "Pharma & Healthcare",
    "HOSPITALS": "Pharma & Healthcare",
    "DRUGS & PHARMACEUTICALS": "Pharma & Healthcare",

    # FMCG
    "FMCG": "FMCG",
    "CONSUMER GOODS": "FMCG",
    "FOOD & BEVERAGES": "FMCG",
    "FOOD PRODUCTS": "FMCG",
    "PERSONAL PRODUCTS": "FMCG",
    "TOBACCO": "FMCG",
    "HOUSEHOLD PRODUCTS": "FMCG",

    # Auto
    "AUTOMOBILE": "Automobile",
    "AUTO": "Automobile",
    "AUTO COMPONENTS": "Automobile",
    "AUTO ANCILLARIES": "Automobile",
    "AUTOMOBILES & AUTO COMPONENTS": "Automobile",
    "COMMERCIAL VEHICLES": "Automobile",
    "2/3 WHEELERS": "Automobile",
    "PASSENGER CARS & UTILITY VEHICLES": "Automobile",

    # Energy & Oil
    "OIL & GAS": "Energy",
    "OIL EXPLORATION": "Energy",
    "OIL MARKETING & DISTRIBUTION": "Energy",
    "REFINERIES": "Energy",
    "GAS": "Energy",
    "PETROLEUM PRODUCTS": "Energy",
    "POWER": "Energy",
    "POWER GENERATION & DISTRIBUTION": "Energy",
    "ELECTRIC UTILITIES": "Energy",

    # Metals & Mining
    "METALS": "Metals & Mining",
    "METALS & MINING": "Metals & Mining",
    "STEEL": "Metals & Mining",
    "IRON & STEEL": "Metals & Mining",
    "ALUMINIUM": "Metals & Mining",
    "ZINC": "Metals & Mining",
    "COPPER": "Metals & Mining",
    "MINING": "Metals & Mining",
    "MINING & MINERAL PRODUCTS": "Metals & Mining",

    # Infrastructure & Construction
    "CONSTRUCTION": "Infrastructure",
    "INFRASTRUCTURE": "Infrastructure",
    "CEMENT": "Infrastructure",
    "CEMENT & CEMENT PRODUCTS": "Infrastructure",
    "CONSTRUCTION MATERIALS": "Infrastructure",
    "REAL ESTATE": "Infrastructure",
    "REALTY": "Infrastructure",

    # Telecom
    "TELECOM": "Telecom",
    "TELECOMMUNICATION": "Telecom",
    "TELECOM - SERVICES": "Telecom",
    "TELECOM EQUIPMENT & ACCESSORIES": "Telecom",

    # Capital Goods
    "CAPITAL GOODS": "Capital Goods",
    "INDUSTRIAL MANUFACTURING": "Capital Goods",
    "ENGINEERING": "Capital Goods",
    "ELECTRICAL EQUIPMENT": "Capital Goods",
    "HEAVY ELECTRICAL EQUIPMENT": "Capital Goods",
    "DEFENCE": "Capital Goods",

    # Chemicals
    "CHEMICALS": "Chemicals",
    "AGROCHEMICALS": "Chemicals",
    "SPECIALTY CHEMICALS": "Chemicals",
    "FERTILIZERS": "Chemicals",
    "PESTICIDES": "Chemicals",
    "PAINTS": "Chemicals",

    # Textiles
    "TEXTILES": "Textiles",
    "TEXTILE": "Textiles",
    "READYMADE GARMENTS/ APPARELS": "Textiles",

    # Media & Entertainment
    "MEDIA": "Media & Entertainment",
    "ENTERTAINMENT": "Media & Entertainment",
    "PRINTING & PUBLISHING": "Media & Entertainment",

    # Consumer Durables
    "CONSUMER DURABLES": "Consumer Durables",
    "ELECTRONICS": "Consumer Durables",
    "HOUSEHOLD APPLIANCES": "Consumer Durables",
}

# Hardcoded sector for top NSE stocks (reliable fallback)
TOP_STOCK_SECTORS: Dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "Information Technology",
    "HDFCBANK": "Banking & Finance",
    "INFY": "Information Technology",
    "ICICIBANK": "Banking & Finance",
    "HINDUNILVR": "FMCG",
    "SBIN": "Banking & Finance",
    "BHARTIARTL": "Telecom",
    "ITC": "FMCG",
    "KOTAKBANK": "Banking & Finance",
    "LT": "Capital Goods",
    "AXISBANK": "Banking & Finance",
    "BAJFINANCE": "Banking & Finance",
    "ASIANPAINT": "Chemicals",
    "MARUTI": "Automobile",
    "TITAN": "Consumer Durables",
    "SUNPHARMA": "Pharma & Healthcare",
    "NESTLEIND": "FMCG",
    "ULTRACEMCO": "Infrastructure",
    "WIPRO": "Information Technology",
    "HCLTECH": "Information Technology",
    "TATAMOTORS": "Automobile",
    "TATASTEEL": "Metals & Mining",
    "POWERGRID": "Energy",
    "NTPC": "Energy",
    "ONGC": "Energy",
    "JSWSTEEL": "Metals & Mining",
    "ADANIENT": "Infrastructure",
    "ADANIPORTS": "Infrastructure",
    "COALINDIA": "Metals & Mining",
    "DIVISLAB": "Pharma & Healthcare",
    "DRREDDY": "Pharma & Healthcare",
    "CIPLA": "Pharma & Healthcare",
    "GRASIM": "Infrastructure",
    "EICHERMOT": "Automobile",
    "BRITANNIA": "FMCG",
    "BAJAJFINSV": "Banking & Finance",
    "BAJAJ-AUTO": "Automobile",
    "HEROMOTOCO": "Automobile",
    "M&M": "Automobile",
    "TECHM": "Information Technology",
    "INDUSINDBK": "Banking & Finance",
    "HDFCLIFE": "Banking & Finance",
    "SBILIFE": "Banking & Finance",
    "DABUR": "FMCG",
    "GODREJCP": "FMCG",
    "PIDILITIND": "Chemicals",
    "HAVELLS": "Consumer Durables",
    "DMART": "FMCG",
    "TATACONSUM": "FMCG",
    "APOLLOHOSP": "Pharma & Healthcare",
    "LUPIN": "Pharma & Healthcare",
    "BIOCON": "Pharma & Healthcare",
    "ZOMATO": "Information Technology",
    "PAYTM": "Information Technology",
    "NYKAA": "FMCG",
    "POLICYBZR": "Banking & Finance",
    "TATAPOWER": "Energy",
    "VEDL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "BPCL": "Energy",
    "IOC": "Energy",
    "GAIL": "Energy",
    "SAIL": "Metals & Mining",
    "BANKBARODA": "Banking & Finance",
    "PNB": "Banking & Finance",
    "CANBK": "Banking & Finance",
    "IDFCFIRSTB": "Banking & Finance",
    "FEDERALBNK": "Banking & Finance",
    "DLF": "Infrastructure",
    "GODREJPROP": "Infrastructure",
    "OBEROIRLTY": "Infrastructure",
    "PRESTIGE": "Infrastructure",
    "ACC": "Infrastructure",
    "AMBUJACEM": "Infrastructure",
    "SHREECEM": "Infrastructure",
    "SIEMENS": "Capital Goods",
    "ABB": "Capital Goods",
    "BEL": "Capital Goods",
    "HAL": "Capital Goods",
    "VOLTAS": "Consumer Durables",
    "WHIRLPOOL": "Consumer Durables",
    "CROMPTON": "Consumer Durables",
    "TVSMOTOR": "Automobile",
    "ASHOKLEY": "Automobile",
    "ESCORTS": "Automobile",
    "BERGEPAINT": "Chemicals",
    "ATUL": "Chemicals",
    "SRF": "Chemicals",
    "DEEPAKNTR": "Chemicals",
    "PIIND": "Chemicals",
    "UPL": "Chemicals",
}


class SectorClassifier:
    """
    Classifies NSE stocks into sectors for concentration monitoring.

    Priority order:
      1. Manual overrides
      2. NSE bhavcopy industry data (if loaded)
      3. Hardcoded top-stock map
      4. "UNCLASSIFIED" fallback
    """

    def __init__(self):
        self._manual_overrides: Dict[str, str] = {}
        self._bhavcopy_sectors: Dict[str, str] = {}
        self._all_sectors: Set[str] = set()
        self._refresh_sectors()

    def _refresh_sectors(self) -> None:
        """Collect all known sector names."""
        self._all_sectors = set(INDUSTRY_TO_SECTOR.values()) | set(TOP_STOCK_SECTORS.values())
        self._all_sectors.add("UNCLASSIFIED")

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        sym = symbol.upper().strip()

        # 1. Manual override
        if sym in self._manual_overrides:
            return self._manual_overrides[sym]

        # 2. Bhavcopy data
        if sym in self._bhavcopy_sectors:
            return self._bhavcopy_sectors[sym]

        # 3. Hardcoded top stocks
        if sym in TOP_STOCK_SECTORS:
            return TOP_STOCK_SECTORS[sym]

        return "UNCLASSIFIED"

    def set_override(self, symbol: str, sector: str) -> None:
        """Manually override sector classification for a symbol."""
        self._manual_overrides[symbol.upper().strip()] = sector

    def load_from_bhavcopy(self, data: Dict[str, str]) -> int:
        """
        Load sector data from NSE bhavcopy INDUSTRY field.

        Args:
            data: {symbol: industry_name} from bhavcopy

        Returns:
            Number of symbols classified
        """
        count = 0
        for sym, industry in data.items():
            sym = sym.upper().strip()
            industry_upper = industry.upper().strip()
            sector = INDUSTRY_TO_SECTOR.get(industry_upper)
            if sector:
                self._bhavcopy_sectors[sym] = sector
                count += 1
            else:
                # Try partial matching
                for key, val in INDUSTRY_TO_SECTOR.items():
                    if key in industry_upper or industry_upper in key:
                        self._bhavcopy_sectors[sym] = val
                        count += 1
                        break
                else:
                    self._bhavcopy_sectors[sym] = "UNCLASSIFIED"

        self._refresh_sectors()
        logger.info("Loaded sector data: %d/%d symbols classified", count, len(data))
        return count

    def get_sector_breakdown(self, positions: List[dict]) -> Dict[str, dict]:
        """
        Get sector breakdown for portfolio positions.

        Args:
            positions: list of {symbol, notional} dicts

        Returns:
            {sector: {notional, count, symbols, pct}} breakdown
        """
        total_notional = sum(p.get("notional", 0) for p in positions)
        sectors: Dict[str, dict] = {}

        for pos in positions:
            symbol = pos.get("symbol", "")
            notional = pos.get("notional", 0)
            sector = self.get_sector(symbol)

            if sector not in sectors:
                sectors[sector] = {"notional": 0, "count": 0, "symbols": [], "pct": 0}
            sectors[sector]["notional"] += notional
            sectors[sector]["count"] += 1
            sectors[sector]["symbols"].append(symbol)

        # Calculate percentages
        if total_notional > 0:
            for s in sectors:
                sectors[s]["pct"] = round(sectors[s]["notional"] / total_notional * 100, 2)

        return sectors

    def check_concentration(
        self,
        positions: List[dict],
        new_symbol: str,
        new_notional: float,
        max_sector_pct: float = 30.0,
    ) -> tuple:
        """
        Check if adding a new position would breach sector concentration limit.

        Returns:
            (allowed: bool, sector: str, current_pct: float, projected_pct: float)
        """
        sector = self.get_sector(new_symbol)
        all_positions = positions + [{"symbol": new_symbol, "notional": new_notional}]
        breakdown = self.get_sector_breakdown(all_positions)

        sector_data = breakdown.get(sector, {"pct": 0})
        projected_pct = sector_data.get("pct", 0)

        # Compute current sector % BEFORE adding the new position
        current_breakdown = self.get_sector_breakdown(positions)
        current_sector_data = current_breakdown.get(sector, {"pct": 0})
        current_pct = current_sector_data.get("pct", 0)
        return projected_pct <= max_sector_pct, sector, current_pct, projected_pct

    def list_sectors(self) -> List[str]:
        """Return all known sectors."""
        return sorted(self._all_sectors)

    def coverage_stats(self, symbols: List[str]) -> dict:
        """Get classification coverage statistics."""
        classified = sum(1 for s in symbols if self.get_sector(s) != "UNCLASSIFIED")
        return {
            "total": len(symbols),
            "classified": classified,
            "unclassified": len(symbols) - classified,
            "coverage_pct": round(classified / len(symbols) * 100, 1) if symbols else 0,
        }
