from datetime import datetime, timedelta
from typing import Optional

import pytz
from dateutil import parser as date_parser
from langchain_core.tools import tool as create_tool
from loguru import logger


@create_tool
def current_time_tool() -> str:
    """
    Get the current time.

    This tool returns the current date and time in ISO 8601 format.
    It can be used to retrieve the current timestamp during reasoning.

    Returns:
        str: The current date and time in ISO 8601 format.
    """

    time_now: str = datetime.now().isoformat()
    logger.trace("Current time retrieved", time=time_now)
    return time_now


@create_tool
def time_in_timezone_tool(timezone_name: str) -> str:
    """
    Get the current time in a specific timezone.

    Args:
        timezone_name: The timezone name (e.g., 'US/Eastern', 'Europe/London', 'Asia/Tokyo')

    Returns:
        str: Current time in the specified timezone with timezone info
    """
    try:
        tz = pytz.timezone(timezone_name)
        time_in_tz = datetime.now(tz)
        logger.trace(
            "Time in timezone retrieved", timezone=timezone_name, time=str(time_in_tz)
        )
        return f"{time_in_tz.strftime('%Y-%m-%d %H:%M:%S %Z')} ({timezone_name})"
    except Exception:
        return f"Error: Invalid timezone '{timezone_name}'. Use format like 'US/Eastern' or 'Europe/London'"


@create_tool
def time_difference_tool(datetime1: str, datetime2: str) -> str:
    """
    Calculate the time difference between two datetime strings.

    Args:
        datetime1: First datetime in ISO format or common formats
        datetime2: Second datetime in ISO format or common formats

    Returns:
        str: Time difference description
    """
    try:
        # Use dateutil parser for flexible datetime parsing
        dt1 = date_parser.parse(datetime1)
        dt2 = date_parser.parse(datetime2)
        diff = abs(dt2 - dt1)

        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 and not parts:  # Only show seconds if no larger units
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        result = ", ".join(parts) if parts else "0 seconds"
        logger.trace(
            "Time difference calculated",
            datetime1=datetime1,
            datetime2=datetime2,
            result=result,
        )
        return result

    except Exception as e:
        return f"Error calculating time difference: {str(e)}"


@create_tool
def add_time_tool(
    base_datetime: str, days: int = 0, hours: int = 0, minutes: int = 0
) -> str:
    """
    Add time to a given datetime.

    Args:
        base_datetime: Base datetime string in various formats (ISO, natural language, etc.)
        days: Number of days to add (can be negative)
        hours: Number of hours to add (can be negative)
        minutes: Number of minutes to add (can be negative)

    Returns:
        str: New datetime after adding the specified time
    """
    try:
        # Parse the base datetime
        if base_datetime.lower() == "now":
            base_dt = datetime.now()
        else:
            # Use dateutil parser for flexible datetime parsing
            base_dt = date_parser.parse(base_datetime)

        # Add the time delta
        new_dt = base_dt + timedelta(days=days, hours=hours, minutes=minutes)

        result = new_dt.isoformat()
        logger.trace(
            "Time added to datetime",
            base_datetime=base_datetime,
            days=days,
            hours=hours,
            minutes=minutes,
            result=result,
        )
        return result

    except Exception as e:
        return f"Error adding time: {str(e)}"


@create_tool
def is_business_hours_tool(
    datetime_str: Optional[str] = None, timezone_name: str = "US/Eastern"
) -> str:
    """
    Check if a given time (or current time) falls within business hours.

    Args:
        datetime_str: Datetime string to check in various formats (defaults to current time)
        timezone_name: Timezone for business hours check

    Returns:
        str: Whether it's business hours and additional context
    """
    try:
        if datetime_str is None:
            check_time = datetime.now()
        else:
            # Use dateutil parser for flexible datetime parsing
            check_time = date_parser.parse(datetime_str)

        # Convert to specified timezone
        tz = pytz.timezone(timezone_name)
        if check_time.tzinfo is None:
            check_time = tz.localize(check_time)
        else:
            check_time = check_time.astimezone(tz)

        # Check if it's a weekday (Monday=0, Sunday=6)
        is_weekday = check_time.weekday() < 5

        # Check if it's between 9 AM and 5 PM
        is_work_hours = 9 <= check_time.hour < 17

        is_business_hours = is_weekday and is_work_hours

        day_name = check_time.strftime("%A")
        time_str = check_time.strftime("%I:%M %p")

        result = f"{'Yes' if is_business_hours else 'No'} - {day_name} at {time_str} {timezone_name}"
        if not is_weekday:
            result += " (Weekend)"
        elif not is_work_hours:
            result += " (Outside 9 AM - 5 PM)"

        logger.trace("Business hours check completed", result=result)
        return result

    except Exception as e:
        return f"Error checking business hours: {str(e)}"


@create_tool
def format_time_tool(datetime_str: str, format_type: str = "readable") -> str:
    """
    Format a datetime string in various human-readable formats.

    Args:
        datetime_str: Datetime string to format in various formats
        format_type: Type of formatting ('readable', 'short', 'long', 'time_only', 'date_only')

    Returns:
        str: Formatted datetime string
    """
    try:
        # Use dateutil parser for flexible datetime parsing
        dt = date_parser.parse(datetime_str)

        formats = {
            "readable": "%B %d, %Y at %I:%M %p",
            "short": "%m/%d/%Y %H:%M",
            "long": "%A, %B %d, %Y at %I:%M:%S %p",
            "time_only": "%I:%M %p",
            "date_only": "%B %d, %Y",
        }

        if format_type not in formats:
            return f"Error: Unknown format type. Use: {', '.join(formats.keys())}"

        result = dt.strftime(formats[format_type])
        logger.trace(
            "Datetime formatted",
            datetime_str=datetime_str,
            format_type=format_type,
            result=result,
        )
        return result

    except Exception as e:
        return f"Error formatting time: {str(e)}"


@create_tool
def time_until_tool(target_datetime: str) -> str:
    """
    Calculate how much time remains until a target datetime.

    Args:
        target_datetime: Target datetime string in various formats

    Returns:
        str: Human-readable time remaining until target
    """
    try:
        now = datetime.now()
        # Use dateutil parser for flexible datetime parsing
        target = date_parser.parse(target_datetime)

        # Remove timezone info for comparison if target has no timezone
        if target.tzinfo is None:
            target = target.replace(tzinfo=None)
        if now.tzinfo is None:
            now = now.replace(tzinfo=None)

        diff = target - now

        if diff.total_seconds() < 0:
            # Target is in the past
            past_diff = abs(diff)
            days = past_diff.days
            hours, remainder = divmod(past_diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

            result = f"Target was {', '.join(parts)} ago"
        else:
            # Target is in the future
            days = diff.days
            hours, remainder = divmod(diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

            result = (
                f"{', '.join(parts)} remaining"
                if parts
                else "Less than 1 minute remaining"
            )

        logger.trace(
            "Time until target calculated",
            target_datetime=target_datetime,
            result=result,
        )
        return result

    except Exception as e:
        return f"Error calculating time until target: {str(e)}"
