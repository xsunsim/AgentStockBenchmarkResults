from __future__ import annotations

import datetime as dt
from typing import Optional, Union


DateLike = Union[str, dt.date, None]


def parse_date(value: DateLike) -> Optional[dt.date]:
    if value is None or isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return dt.datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 10:
        return dt.date.fromisoformat(text)
    raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {value!r}")


def require_date(value: DateLike) -> dt.date:
    parsed = parse_date(value)
    if parsed is None:
        raise ValueError("date is required")
    return parsed


def date_id(value: DateLike) -> Optional[str]:
    parsed = parse_date(value)
    return parsed.strftime("%Y%m%d") if parsed else None
