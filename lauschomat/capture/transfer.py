"""
File transfer functionality for sending recordings to the GPU server.
"""
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

from lauschomat.common.config import Config

logger = logging.getLogger(__name__)


class FileTransferManager:
    """Manages transfer of files to the GPU server."""
    
    def __init__(self, config: Config):
        """Initialize file transfer manager."""
        self.config = config
        if not config.transfer:
            raise ValueError("Transfer configuration is required")
        
        self.transfer_config = config.transfer
        self.data_root = Path(config.app.data_root)
        
        # Queue for files to transfer
        self.queue = queue.Queue()
        
        # Set of files currently being transferred
        self.in_progress: Set[Path] = set()
        
        # Set of files that have been transferred
        self.transferred: Set[Path] = set()
        
        # Thread for processing queue
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure necessary directories exist."""
        transfer_dir = self.data_root / "transfer"
        queue_dir = transfer_dir / "queue"
        sent_dir = transfer_dir / "sent"
        
        os.makedirs(queue_dir, exist_ok=True)
        os.makedirs(sent_dir, exist_ok=True)
    
    def queue_file(self, file_path: Path) -> bool:
        """Queue a file for transfer."""
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        if file_path in self.transferred:
            logger.debug(f"File already transferred: {file_path}")
            return True
        
        if file_path in self.in_progress:
            logger.debug(f"File already in progress: {file_path}")
            return True
        
        try:
            # Add to queue
            self.queue.put(file_path)
            logger.debug(f"Queued file for transfer: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to queue file for transfer: {e}")
            return False
    
    def start(self):
        """Start the transfer manager."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        logger.info("File transfer manager started")
    
    def stop(self):
        """Stop the transfer manager."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("File transfer manager stopped")
    
    def _process_queue(self):
        """Process the queue of files to transfer."""
        while self.running:
            try:
                # Get file from queue with timeout
                try:
                    file_path = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Mark as in progress
                self.in_progress.add(file_path)
                
                # Transfer file
                success = self._transfer_file(file_path)
                
                # Update status
                if success:
                    self.transferred.add(file_path)
                    # Record in sent directory
                    self._record_sent(file_path)
                
                # Remove from in progress
                self.in_progress.remove(file_path)
                
                # Mark task as done
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error processing transfer queue: {e}")
                time.sleep(1.0)  # Avoid tight loop on error
    
    def _transfer_file(self, file_path: Path) -> bool:
        """Transfer a file to the GPU server."""
        if not file_path.exists():
            logger.warning(f"File no longer exists: {file_path}")
            return False
        
        method = self.transfer_config.method.lower()
        
        try:
            if method == "remote_fs":
                return self._transfer_remote_fs(file_path)
            elif method == "rsync":
                return self._transfer_rsync(file_path)
            elif method == "scp":
                return self._transfer_scp(file_path)
            else:
                logger.error(f"Unsupported transfer method: {method}")
                return False
        except Exception as e:
            logger.error(f"Error transferring file {file_path}: {e}")
            return False
    
    def _transfer_remote_fs(self, file_path: Path) -> bool:
        """Transfer file using remote filesystem API."""
        # This is a placeholder for the actual remote filesystem API
        # In production, this would use the specific API for your environment
        logger.info(f"Would transfer {file_path} via remote filesystem API")
        
        # For development/testing, simulate success
        time.sleep(0.1)  # Simulate transfer time
        return True
    
    def _transfer_rsync(self, file_path: Path) -> bool:
        """Transfer file using rsync."""
        target = f"{self.transfer_config.target_host}:{self.transfer_config.target_path}"
        
        # Prepare command
        cmd = [
            "rsync",
            "-avz",  # Archive mode, verbose, compress
            "--progress",
            str(file_path),
            target
        ]
        
        # Add SSH key if specified
        if self.transfer_config.ssh_key_path:
            ssh_key = os.path.expanduser(self.transfer_config.ssh_key_path)
            cmd.insert(1, f"-e ssh -i {ssh_key}")
        
        # Execute command
        try:
            logger.debug(f"Running rsync command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"rsync output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"rsync failed: {e.stderr}")
            return False
    
    def _transfer_scp(self, file_path: Path) -> bool:
        """Transfer file using scp."""
        target = f"{self.transfer_config.target_host}:{self.transfer_config.target_path}"
        
        # Prepare command
        cmd = [
            "scp",
            str(file_path),
            target
        ]
        
        # Add SSH key if specified
        if self.transfer_config.ssh_key_path:
            ssh_key = os.path.expanduser(self.transfer_config.ssh_key_path)
            cmd.insert(1, f"-i {ssh_key}")
        
        # Execute command
        try:
            logger.debug(f"Running scp command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"scp output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"scp failed: {e.stderr}")
            return False
    
    def _record_sent(self, file_path: Path):
        """Record that a file has been sent."""
        sent_dir = self.data_root / "transfer" / "sent"
        
        # Create a marker file
        marker = sent_dir / f"{file_path.name}.sent"
        with open(marker, 'w') as f:
            f.write(f"{time.time()}\n")
            f.write(f"{file_path}\n")
