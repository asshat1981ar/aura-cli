class ExceptionHandler:
    """Standardized exception handler to reduce duplicate code and improve reuse."""
    
    def __init__(self):
        self.exception_patterns = {}
    
    def register_pattern(self, exception_type, handler):
        """Register a reusable exception handling pattern."""
        self.exception_patterns[exception_type] = handler
    
    def handle(self, exception):
        """Handle an exception using registered patterns."""
        handler = self.exception_patterns.get(type(exception))
        if handler:
            return handler(exception)
        else:
            raise exception

def common_io_handler(exception):
    """Standard handler for common I/O exceptions."""
    print(f"I/O Error occurred: {exception}")
    return None

def common_value_handler(exception):
    """Standard handler for common value-related exceptions."""
    print(f"Value Error occurred: {exception}")
    return None

# Initialize default handlers
default_handler = ExceptionHandler()
default_handler.register_pattern(IOError, common_io_handler)
default_handler.register_pattern(ValueError, common_value_handler)