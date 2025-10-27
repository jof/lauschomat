"""
File watcher for detecting new audio files.
"""
import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, List, Set

logger = logging.getLogger(__name__)


class FileWatcher:
    """Watches a directory for new files matching patterns."""
    
    def __init__(self, watch_dir: Path, patterns: List[str], callback: Callable[[Path], None], interval_sec: float = 1.0):
        """Initialize file watcher.
        
        Args:
            watch_dir: Directory to watch
            patterns: List of glob patterns to match
            callback: Function to call when a new file is found
            interval_sec: Polling interval in seconds
        """
        self.watch_dir = watch_dir
        self.patterns = patterns
        self.callback = callback
        self.interval_sec = interval_sec
        self.known_files: Set[Path] = set()
        self.thread: threading.Thread = None
        self.running = False
    
    def start(self):
        """Start watching for files."""
        if self.running:
            return
        
        # Ensure directory exists
        os.makedirs(self.watch_dir, exist_ok=True)
        
        # Scan for existing files
        self._scan_initial()
        
        # Start watching thread
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info(f"File watcher started on {self.watch_dir}")
    
    def stop(self):
        """Stop watching for files."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("File watcher stopped")
    
    def _scan_initial(self):
        """Scan for existing files."""
        try:
            for file_path in self.watch_dir.glob("**/*"):
                if file_path.is_file() and self._matches_patterns(file_path):
                    self.known_files.add(file_path)
        except Exception as e:
            logger.error(f"Error scanning initial files: {e}")
    
    def _watch_loop(self):
        """Main watching loop."""
        while self.running:
            try:
                # Scan for new files
                self._scan_for_new_files()
                
                # Sleep
                time.sleep(self.interval_sec)
            except Exception as e:
                logger.error(f"Error in file watcher: {e}")
                time.sleep(self.interval_sec)  # Avoid tight loop on error
    
    def _scan_for_new_files(self):
        """Scan for new files and call callback."""
        new_files = set()
        
        try:
            # Find all matching files
            for file_path in self.watch_dir.glob("**/*"):
                if file_path.is_file() and self._matches_patterns(file_path):
                    new_files.add(file_path)
            
            # Find files that are new
            added_files = new_files - self.known_files
            
            # Update known files
            self.known_files = new_files
            
            # Call callback for new files
            for file_path in added_files:
                try:
                    self.callback(file_path)
                except Exception as e:
                    logger.error(f"Error in file callback: {e}")
        except Exception as e:
            logger.error(f"Error scanning for new files: {e}")
    
    def _matches_patterns(self, file_path: Path) -> bool:
        """Check if a file matches any of the patterns."""
        file_name = file_path.name
        return any(fnmatch.fnmatch(file_name, pattern) for pattern in self.patterns)
