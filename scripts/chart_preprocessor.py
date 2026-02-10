"""
Shared chart preprocessing module — converts computation engine JSON to compact YAML.

This is the SINGLE SOURCE OF TRUTH for chart preprocessing. Used by:
  - 09_chat_ui.py (Gradio WebUI)
  - 11_api_server.py (FastAPI)
  - 13_generate_dpo_dataset.py (DPO training data generation)

Why YAML? ~3x more token-efficient than JSON for LLMs.
A 5500-line JSON chart becomes ~200 lines of YAML — fits comfortably in 32k context.

Industry principle: training data format MUST match inference format.
The model must see the EXACT same YAML structure during DPO training
as it will see during live inference.
"""

import json
from datetime import date, datetime


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_CHART_YAML = 14000  # token budget for chart YAML (increased for pratyantar dashas)

# Planet lord abbreviation → full name mapping (computation engine uses 3-letter codes)
_LORD_MAP = {
    "SUN": "Sun", "MON": "Moon", "MER": "Mercury", "VEN": "Venus",
    "MAR": "Mars", "JUP": "Jupiter", "SAT": "Saturn", "RAH": "Rahu", "KET": "Ketu",
}

# Month abbreviation map for human-readable dates
_MONTH_ABBR = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

# 9 traditional KP planets (exclude Uranus, Neptune, Pluto)
_KP_PLANETS = ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Rahu", "Ketu"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _iso_to_readable(iso_str: str) -> str:
    """Convert '2025-10-22T...' or '2025-10-22' to 'Oct 2025'. Return '?' on failure."""
    if not iso_str or len(iso_str) < 7:
        return "?"
    try:
        parts = iso_str[:10].split("-")
        return f"{_MONTH_ABBR.get(parts[1], parts[1])} {parts[0]}"
    except Exception:
        return iso_str[:7]


def _lord_full(abbr: str) -> str:
    """Convert 3-letter lord abbreviation to full name. E.g. 'VEN' -> 'Venus'."""
    return _LORD_MAP.get(abbr, abbr)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION: chart_to_yaml
# ═══════════════════════════════════════════════════════════════════════════════

def chart_to_yaml(raw: str) -> str:
    """Convert computation-engine JSON (~5500 lines) to compact YAML (~200 lines).

    YAML is ~3x more token-efficient than JSON for LLMs.
    Keeps: sign/star/sub for planets+cusps, significators, planet significations,
    current+next mahadasha with antardashas AND pratyantar dashas for precision.
    Uses human-readable dates (Oct 2025 instead of 2025-10).
    Includes today's date for past/future awareness.

    Args:
        raw: Either a JSON string from computation engine, or pre-formatted YAML string.

    Returns:
        Compact YAML string ready for LLM consumption.
    """
    raw = raw.strip()
    if not raw:
        return ""

    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Already YAML or other format — return as-is (truncated if needed)
        if len(raw) > MAX_CHART_YAML:
            return raw[:MAX_CHART_YAML] + "\n# ...truncated"
        return raw

    lines = []
    today = date.today()
    today_str = today.isoformat()

    # ── Context: today's date ──
    lines.append(f"today_date: {today.strftime('%d %b %Y')}")

    # ── Native info ──
    name = d.get("name", "Unknown")
    gender = d.get("gender", "")
    bd = d.get("birthDetails", {})
    lines.append("native:")
    lines.append(f"  name: \"{name}\"")
    if gender:
        lines.append(f"  gender: {gender}")
    lines.append(f"  dob: \"{bd.get('date', '?')}\"")
    lines.append(f"  tob: \"{bd.get('time', '?')}\"")
    if bd.get("place"):
        lines.append(f"  place: \"{bd.get('place', '')}\"")
    lines.append(f"  lagna: {bd.get('lagna', '?')} ({_lord_full(bd.get('lagnaLord', '?'))})")
    lines.append(f"  rasi: {bd.get('rasi', '?')} ({_lord_full(bd.get('rasiLord', '?'))})")
    lines.append(f"  nakshatra: {bd.get('nakshatra', '?')} ({_lord_full(bd.get('nakshatraLord', '?'))})")

    # ── Calculate age for age-aware responses ──
    dob = None
    try:
        dob_str = bd.get("date", "")
        if "." in dob_str:
            dob = datetime.strptime(dob_str, "%d.%m.%Y").date()
        elif "-" in dob_str:
            dob = datetime.strptime(dob_str, "%d-%m-%Y").date()
        if dob:
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            lines.append(f"  age_now: {age}")
    except Exception:
        pass

    # ── Planet KP (sign | star | sub + significations) ──
    pkp = d.get("planetKP", {})
    psig = d.get("planetSignifications", {})
    if pkp:
        lines.append("")
        lines.append("planets:")
        for planet in _KP_PLANETS:
            p = pkp.get(planet, {})
            if not p:
                continue
            houses = psig.get(planet, [])
            house_str = f" | houses: {houses}" if houses else ""
            lines.append(
                f"  {planet}: {p.get('rashi','?')} | {p.get('nakshatra','?')} ({_lord_full(p.get('nakshatraLord','?'))}) "
                f"| sub: {_lord_full(p.get('sub','?'))}{house_str}"
            )

    # ── Cusp KP (sign | star | sub) — all 12 cusps ──
    ckp = d.get("cuspKP", {})
    if ckp:
        lines.append("")
        lines.append("cusps:")
        for c in [str(i) for i in range(1, 13)]:
            cp = ckp.get(c, {})
            if cp:
                lines.append(
                    f"  {c}: {cp.get('rashi','?')} | {cp.get('nakshatra','?')} ({_lord_full(cp.get('nakshatraLord','?'))}) "
                    f"| sub: {_lord_full(cp.get('sub','?'))}"
                )

    # ── Significators ──
    sig = d.get("significators", {})
    if sig:
        lines.append("")
        lines.append("significators:")
        for house in [str(i) for i in range(1, 13)]:
            planets = sig.get(house, [])
            if planets:
                full_names = [_lord_full(p) for p in planets]
                lines.append(f"  {house}: [{', '.join(full_names)}]")

    # ── Dashas — with PRATYANTAR dashas for week-to-month precision ──
    dashas_data = d.get("dashas", {})
    db = dashas_data.get("dashaBalance", {})
    dlist = dashas_data.get("dashas", [])

    if db or dlist:
        lines.append("")
        lines.append("dashas:")
        if db:
            lines.append(
                f"  balance: {_lord_full(db.get('lord','?'))} "
                f"{db.get('years',0)}Y {db.get('months',0)}M {db.get('days',0)}D remaining"
            )

        if dlist:
            # Find current mahadasha
            current_md_idx = 0
            for i, dd in enumerate(dlist):
                start = dd.get("startDate", "")[:10]
                end = dd.get("endDate", "")[:10]
                if start <= today_str <= end:
                    current_md_idx = i
                    break

            # Include previous mahadasha (last 4 antardashas) for past-event queries
            if current_md_idx > 0:
                prev_md = dlist[current_md_idx - 1]
                lord = _lord_full(prev_md.get("lord", "?"))
                lines.append(f"  # Previous mahadasha (for past events)")
                lines.append(f"  {lord}_mahadasha: {_iso_to_readable(prev_md.get('startDate',''))} to {_iso_to_readable(prev_md.get('endDate',''))}")
                prev_antars = prev_md.get("antarDashas", [])
                if prev_antars:
                    lines.append(f"    antardashas (last 4):")
                    for ad in prev_antars[-4:]:
                        ad_lord = _lord_full(ad.get("lord", "?"))
                        lines.append(f"      {lord}-{ad_lord}: {_iso_to_readable(ad.get('startDate',''))} to {_iso_to_readable(ad.get('endDate',''))}")

            # Current + next mahadasha with full antardashas + pratyantar dashas
            for md_offset, dd in enumerate(dlist[current_md_idx:current_md_idx + 2]):
                lord = _lord_full(dd.get("lord", "?"))
                md_start = _iso_to_readable(dd.get("startDate", ""))
                md_end = _iso_to_readable(dd.get("endDate", ""))
                period = dd.get("period", "?")
                label = "# Current mahadasha" if md_offset == 0 else "# Next mahadasha"
                lines.append(f"  {label}")
                lines.append(f"  {lord}_mahadasha: {md_start} to {md_end} ({period})")

                antars = dd.get("antarDashas", [])
                if antars:
                    # Find current antardasha
                    current_ad_idx = 0
                    for ai, ad in enumerate(antars):
                        ad_start = ad.get("startDate", "")[:10]
                        ad_end = ad.get("endDate", "")[:10]
                        if ad_start <= today_str <= ad_end:
                            current_ad_idx = ai
                            break

                    lines.append(f"    antardashas:")
                    for ai, ad in enumerate(antars):
                        ad_lord = _lord_full(ad.get("lord", "?"))
                        ad_start_r = _iso_to_readable(ad.get("startDate", ""))
                        ad_end_r = _iso_to_readable(ad.get("endDate", ""))
                        is_current = (ad.get("startDate", "")[:10] <= today_str <= ad.get("endDate", "")[:10])
                        marker = " ← CURRENT" if is_current else ""
                        lines.append(f"      {lord}-{ad_lord}: {ad_start_r} to {ad_end_r}{marker}")

                        # Include pratyantar dashas for:
                        # - current antardasha (always)
                        # - next 2 antardashas in current mahadasha
                        # - first antardasha of next mahadasha
                        include_pratyantar = False
                        if md_offset == 0:  # current mahadasha
                            if is_current or (ai > current_ad_idx and ai <= current_ad_idx + 2):
                                include_pratyantar = True
                        elif md_offset == 1 and ai == 0:  # first AD of next mahadasha
                            include_pratyantar = True

                        pds = ad.get("pratyantarDashas", [])
                        if include_pratyantar and pds:
                            lines.append(f"        pratyantars:")
                            for pd in pds:
                                pd_lord = _lord_full(pd.get("lord", "?"))
                                pd_start = _iso_to_readable(pd.get("startDate", ""))
                                pd_end = _iso_to_readable(pd.get("endDate", ""))
                                pd_is_current = (pd.get("startDate", "")[:10] <= today_str <= pd.get("endDate", "")[:10])
                                pd_marker = " ← NOW" if pd_is_current else ""
                                lines.append(f"          {lord}-{ad_lord}-{pd_lord}: {pd_start} to {pd_end}{pd_marker}")

    # ── Derived significator groups (convenience for the model) ──
    sig = d.get("significators", {})
    if sig:
        marriage_houses = ["2", "7", "11"]
        career_houses = ["2", "6", "10"]
        finance_houses = ["2", "6", "11"]

        def _sig_for(houses):
            result = set()
            for h in houses:
                for p in sig.get(h, []):
                    result.add(_lord_full(p))
            return sorted(result)

        lines.append("")
        lines.append(f"Significators_for_Marriage (houses 2,7,11): {', '.join(_sig_for(marriage_houses))}")
        lines.append(f"Significators_for_Career (houses 2,6,10): {', '.join(_sig_for(career_houses))}")
        lines.append(f"Significators_for_Finance (houses 2,6,11): {', '.join(_sig_for(finance_houses))}")

    result = "\n".join(lines)
    if len(result) > MAX_CHART_YAML:
        result = result[:MAX_CHART_YAML] + "\n# ...truncated"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: Load and preprocess kundali JSON files
# ═══════════════════════════════════════════════════════════════════════════════

def load_kundali_json(filepath: str) -> dict:
    """Load a kundali JSON file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def kundali_to_yaml(filepath: str) -> str:
    """Load a kundali JSON and convert to YAML in one step."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return chart_to_yaml(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: Test the preprocessor on a kundali file
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chart_preprocessor.py <kundali.json>")
        print("  Converts computation engine JSON to compact YAML for LLM consumption.")
        sys.exit(1)

    filepath = sys.argv[1]
    yaml_output = kundali_to_yaml(filepath)
    print(yaml_output)
    print(f"\n# --- Stats ---")
    print(f"# Lines: {len(yaml_output.splitlines())}")
    print(f"# Characters: {len(yaml_output)}")
