"""how money and times are written for the customer: rupees, and india time."""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def rupees(paise: int) -> str:
    whole, rest = divmod(abs(paise), 100)
    text = f"₹{whole:,}" + (f".{rest:02d}" if rest else "")
    return "-" + text if paise < 0 else text


def clock(at: datetime) -> str:
    local = at.astimezone(IST)
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def day(at: datetime) -> str:
    local = at.astimezone(IST)
    return f"{local.day} {MONTHS[local.month - 1]}"


def day_with_year(at: datetime) -> str:
    return f"{day(at)} {at.astimezone(IST).year}"


def when(at: datetime, now: datetime) -> str:
    """the time alone when it is today, otherwise the day too."""
    if at.astimezone(IST).date() == now.astimezone(IST).date():
        return clock(at)
    return f"{clock(at)} on {day(at)}"


def minutes_between(earlier: datetime, later: datetime) -> int:
    return max(0, round((later - earlier).total_seconds() / 60))
