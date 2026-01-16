class ChatbotError(Exception):
    """Base exception for all chatbot related errors"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(ChatbotError):
    """Raised when configuration is missing or invalid"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)

class ModelError(ChatbotError):
    """Raised when AI/Embedding model fails"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "MODEL_ERROR", details)

class ContextError(ChatbotError):
    """Raised when context operations fail"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONTEXT_ERROR", details)

class ProcessingError(ChatbotError):
    """Raised during request processing/routing"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "PROCESSING_ERROR", details)
