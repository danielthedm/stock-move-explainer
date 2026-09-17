from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC timestamp (what we store in the DB)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
