"""
Verbose logger utility for controlling print output based on verbosity level.

Verbosity Levels:
    0 = Silent: No output except errors
    1 = Minimal: Only essential information (iterations, final results)
    2 = Standard: Standard progress information (default)
    3 = Debug: All debug information including optimization steps
"""


class VerboseLogger:
    """Logger with configurable verbosity level."""
    
    # Verbosity level constants
    SILENT = 0
    MINIMAL = 1
    STANDARD = 2
    DEBUG = 3
    
    def __init__(self, level=STANDARD):
        """
        Initialize logger with specified verbosity level.
        
        Args:
            level (int): Verbosity level (0-3)
        """
        self.level = max(0, min(3, level))  # Clamp to 0-3
    
    def set_level(self, level):
        """Set the verbosity level."""
        self.level = max(0, min(3, level))
    
    def silent(self, *args, **kwargs):
        """Print only if level >= SILENT (always, for errors)."""
        if self.level >= self.SILENT:
            print(*args, **kwargs)
    
    def minimal(self, *args, **kwargs):
        """Print only if level >= MINIMAL."""
        if self.level >= self.MINIMAL:
            print(*args, **kwargs)
    
    def standard(self, *args, **kwargs):
        """Print only if level >= STANDARD."""
        if self.level >= self.STANDARD:
            print(*args, **kwargs)
    
    def debug(self, *args, **kwargs):
        """Print only if level >= DEBUG."""
        if self.level >= self.DEBUG:
            print(*args, **kwargs)
    
    def error(self, *args, **kwargs):
        """Always print errors regardless of verbosity level."""
        print(*args, **kwargs)


# Global logger instance
_global_logger = VerboseLogger()


def get_logger():
    """Get the global logger instance."""
    return _global_logger


def set_verbose_level(level):
    """Set the global logger verbosity level."""
    _global_logger.set_level(level)
