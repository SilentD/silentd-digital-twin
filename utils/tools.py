from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def estimate_tokens(text: str) -> int:
    """
    Fallback character-ratio estimator (~4 chars per token) if 
    API provider metadata is not directly returned.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)

def calculate_session_token_metrics(messages: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Calculates inbound and outbound token totals aggregated across:
    - Last 1 Minute
    - Last 1 Hour
    - Last 24 Hours
    - Lifetime Total
    """
    now = datetime.now()
    one_min_ago = now - timedelta(minutes=1)
    one_hour_ago = now - timedelta(hours=1)
    one_day_ago = now - timedelta(days=1)

    metrics = {
        "per_minute": {"inbound": 0, "outbound": 0},
        "per_hour":   {"inbound": 0, "outbound": 0},
        "per_day":    {"inbound": 0, "outbound": 0},
        "total":      {"inbound": 0, "outbound": 0}
    }

    for msg in messages:
        ts_str = msg.get("timestamp")
        usage = msg.get("usage", {})
        if not ts_str or not usage:
            continue

        try:
            msg_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        in_tok = usage.get("inbound_tokens", 0)
        out_tok = usage.get("outbound_tokens", 0)

        # Total
        metrics["total"]["inbound"] += in_tok
        metrics["total"]["outbound"] += out_tok

        # Last 24 Hours
        if msg_time >= one_day_ago:
            metrics["per_day"]["inbound"] += in_tok
            metrics["per_day"]["outbound"] += out_tok

        # Last 1 Hour
        if msg_time >= one_hour_ago:
            metrics["per_hour"]["inbound"] += in_tok
            metrics["per_hour"]["outbound"] += out_tok

        # Last 1 Minute
        if msg_time >= one_min_ago:
            metrics["per_minute"]["inbound"] += in_tok
            metrics["per_minute"]["outbound"] += out_tok

    return metrics

def get_current_timestamp() -> str:
    """Helper tool providing real-time system clock values."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_database_status(vector_db: dict) -> str:
    """Provides details on loaded context entries."""
    if not vector_db:
        return "Knowledge database uninitialized."
    return f"Database contains {len(vector_db['chunks'])} active JSON facts."
