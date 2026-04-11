"""Cron expression parser for task scheduling."""

from datetime import datetime, timedelta
from typing import Optional


class CronExpression:
    """Parse and evaluate cron expressions."""
    
    # Field ranges: minute, hour, day of month, month, day of week
    RANGES = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day of month
        (1, 12),   # month
        (0, 6),    # day of week (0 = Sunday)
    ]
    
    def __init__(self, expression: str):
        """
        Parse a cron expression.
        
        Format: minute hour day_of_month month day_of_week
        Example: "0 2 * * *" = 2:00 AM every day
        """
        self.expression = expression
        self.fields = self._parse(expression)
    
    def _parse(self, expression: str) -> list:
        """Parse cron expression into fields."""
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")
        
        fields = []
        for i, part in enumerate(parts):
            field_values = self._parse_field(part, self.RANGES[i])
            fields.append(field_values)
        
        return fields
    
    def _parse_field(self, field: str, range_tuple: tuple) -> set:
        """Parse a single cron field."""
        values = set()
        min_val, max_val = range_tuple
        
        # Handle comma-separated values
        for segment in field.split(','):
            # Handle ranges
            if '/' in segment:
                # Step value: */5 or 1-10/2
                range_part, step = segment.split('/')
                step = int(step)
                if range_part == '*':
                    start, end = min_val, max_val
                elif '-' in range_part:
                    start, end = map(int, range_part.split('-'))
                else:
                    start = int(range_part)
                    end = max_val
                
                for v in range(start, end + 1, step):
                    if min_val <= v <= max_val:
                        values.add(v)
            
            elif '-' in segment:
                # Range: 1-5
                start, end = map(int, segment.split('-'))
                for v in range(start, end + 1):
                    if min_val <= v <= max_val:
                        values.add(v)
            
            elif segment == '*':
                # All values
                values.update(range(min_val, max_val + 1))
            
            else:
                # Single value
                v = int(segment)
                if min_val <= v <= max_val:
                    values.add(v)
        
        return values
    
    def matches(self, dt: Optional[datetime] = None) -> bool:
        """Check if the given datetime matches the cron expression."""
        if dt is None:
            dt = datetime.utcnow()
        
        # Cron uses 0=Sunday, but Python's weekday() uses 0=Monday
        # Convert Python weekday to cron format
        cron_weekday = (dt.weekday() + 1) % 7
        
        return (
            dt.minute in self.fields[0] and
            dt.hour in self.fields[1] and
            dt.day in self.fields[2] and
            dt.month in self.fields[3] and
            cron_weekday in self.fields[4]
        )
    
    def next_run(self, after: Optional[datetime] = None) -> Optional[datetime]:
        """Get the next run time after the given datetime."""
        if after is None:
            after = datetime.utcnow()
        
        # Start from next minute
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Limit search to 4 years to avoid infinite loops
        max_search = current + timedelta(days=1461)
        
        while current < max_search:
            if self.matches(current):
                return current
            current += timedelta(minutes=1)
        
        return None
    
    def __str__(self) -> str:
        return self.expression


def parse_cron(expression: str) -> CronExpression:
    """Parse a cron expression."""
    return CronExpression(expression)


def is_valid_cron(expression: str) -> bool:
    """Check if a cron expression is valid."""
    try:
        CronExpression(expression)
        return True
    except ValueError:
        return False
