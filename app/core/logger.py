import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        # Add extra fields if available
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

def setup_logger(name: str, level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """
    Setup a logger with console handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        # Standard readable format for dev
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

# Global instance for easy import
system_logger = setup_logger("hybrid_bot", level="INFO")
