import os
import hashlib
import threading
import time
import tempfile
import shutil
from typing import Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AtomicConfigWatcher:
    """
    Watch config files with atomic read protection.
    """
    
    def __init__(self,
                 config_paths: Dict[str, str],
                 on_action_change: Callable[[str], None],
                 on_rule_change: Callable[[str], None],
                 poll_interval: float = 2.0,
                 max_retries: int = 3,
                 retry_delay: float = 0.5):
        self.config_paths = config_paths
        self.on_action_change = on_action_change
        self.on_rule_change = on_rule_change
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self._file_hashes: Dict[str, str] = {}
        self._file_sizes: Dict[str, int] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start watching in background thread."""
        if self._running:
            return
        
        self._running = True
        self._init_hashes()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Config watcher started")
    
    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Config watcher stopped")
    
    def _init_hashes(self):
        """Initialize file hashes."""
        for name, path in self.config_paths.items():
            hash_val, size = self._compute_hash_safe(path)
            self._file_hashes[name] = hash_val
            self._file_sizes[name] = size
    
    def _compute_hash_safe(self, path: str) -> tuple[str, int]:
        """
        Compute file hash with atomic read protection.
        """
        if not os.path.exists(path):
            return "", 0
        
        for attempt in range(self.max_retries):
            try:
                # Check size
                size1 = os.path.getsize(path)
                time.sleep(0.05)  # Small delay
                size2 = os.path.getsize(path)
                
                if size1 != size2:
                    # File is being written
                    logger.debug(f"File {path} is being written, retry {attempt + 1}")
                    time.sleep(self.retry_delay)
                    continue
                
                # Read and hash
                with open(path, 'rb') as f:
                    content = f.read()
                    return hashlib.md5(content).hexdigest(), len(content)
                    
            except (IOError, OSError) as e:
                logger.warning(f"Error reading {path}: {e}, retry {attempt + 1}")
                time.sleep(self.retry_delay)
        
        # Failed after retries
        logger.error(f"Failed to read {path} after {self.max_retries} retries")
        return self._file_hashes.get(os.path.basename(path), ""), 0
    
    def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            time.sleep(self.poll_interval)
            
            for name, path in self.config_paths.items():
                try:
                    new_hash, new_size = self._compute_hash_safe(path)
                    old_hash = self._file_hashes.get(name, "")
                    
                    if new_hash and new_hash != old_hash:
                        logger.info(f"Config changed: {name}")
                        self._file_hashes[name] = new_hash
                        self._file_sizes[name] = new_size
                        self._handle_change(name, path)
                        
                except Exception as e:
                    logger.error(f"Error watching {name}: {e}")
    
    def _handle_change(self, name: str, path: str):
        """Handle config file change."""
        try:
            if "action" in name.lower():
                self.on_action_change(path)
            elif "rule" in name.lower():
                self.on_rule_change(path)
        except Exception as e:
            logger.error(f"Error handling change for {name}: {e}")
