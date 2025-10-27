"""
Web server for visualizing audio recordings and transcriptions.
"""
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lauschomat.common.config import Config

logger = logging.getLogger(__name__)


class WebServer:
    """Web server for visualizing audio recordings and transcriptions."""
    
    def __init__(self, config: Config):
        """Initialize web server."""
        self.config = config
        
        if not config.web:
            raise ValueError("Web configuration is required")
        
        self.web_config = config.web
        self.data_root = Path(config.app.data_root)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Lauschomat",
            description="Audio Recording and Transcription Visualization",
            version="0.1.0",
        )
        
        # Set up templates
        template_dir = self.web_config.template_dir
        if not os.path.exists(template_dir):
            # Use default templates
            template_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=template_dir)
        
        # Set up static files
        static_dir = self.web_config.static_dir
        if not os.path.exists(static_dir):
            # Use default static files
            static_dir = Path(__file__).parent / "static"
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Set up routes
        self._setup_routes()
        
        # Server instance
        self.server = None
        self.thread = None
    
    def _setup_routes(self):
        """Set up API routes."""
        app = self.app
        
        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Render index page."""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "Lauschomat"}
            )
        
        @app.get("/api/transmissions")
        async def get_transmissions(
            date: Optional[str] = None,
            limit: int = Query(100, ge=1, le=1000),
            offset: int = Query(0, ge=0),
        ):
            """Get list of transmissions using directory listing instead of index files."""
            try:
                if not date:
                    # Use today's date
                    date = datetime.now().strftime("%Y-%m-%d")
                
                # Find recordings directory for the date
                recordings_dir = self.data_root / "recordings" / date
                
                if not recordings_dir.exists() or not recordings_dir.is_dir():
                    return {"transmissions": [], "total": 0, "date": date}
                
                # Get all WAV files in the directory
                wav_files = list(recordings_dir.glob("*.wav"))
                
                # Create transmission objects from files
                transmissions = []
                for wav_file in wav_files:
                    # Get base ID (filename without extension)
                    file_id = wav_file.stem
                    
                    # Check for metadata and transcript files
                    meta_file = wav_file.with_suffix(".meta.json")
                    transcript_file = wav_file.with_suffix(".transcript.json")
                    
                    # Skip if metadata doesn't exist (incomplete recording)
                    if not meta_file.exists():
                        continue
                    
                    # Load metadata
                    try:
                        with open(meta_file, "r") as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading metadata for {wav_file}: {e}")
                        continue
                    
                    # Create transmission object
                    transmission = {
                        "id": file_id,
                        "date": date,
                        "timestamp_utc": metadata.get("timestamp_utc", ""),
                        "audio_path": str(wav_file.relative_to(self.data_root)),
                        "metadata_path": str(meta_file.relative_to(self.data_root)),
                        "duration_sec": metadata.get("duration_sec", 0),
                    }
                    
                    # Add transcription path if available
                    if transcript_file.exists():
                        transmission["transcription_path"] = str(transcript_file.relative_to(self.data_root))
                        
                        # Add transcription text if available
                        try:
                            with open(transcript_file, "r") as f:
                                transcript_data = json.load(f)
                                transmission["text"] = transcript_data.get("text", "")
                        except Exception as e:
                            logger.error(f"Error loading transcript for {wav_file}: {e}")
                    
                    transmissions.append(transmission)
                
                # Sort by timestamp (newest first)
                transmissions.sort(key=lambda x: x.get("timestamp_utc", ""), reverse=True)
                
                # Apply pagination
                paginated = transmissions[offset:offset + limit]
                
                return {
                    "transmissions": paginated,
                    "total": len(transmissions),
                    "date": date
                }
            except Exception as e:
                logger.error(f"Error getting transmissions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/transmission/{id}")
        async def get_transmission(id: str):
            """Get a specific transmission by ID using direct file access."""
            try:
                # Extract date from ID (format: YYYYMMDDTHHMMSS_XXXX)
                if len(id) >= 8:
                    date_part = id[:8]  # Extract YYYYMMDD
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    date_formatted = f"{year}-{month}-{day}"
                else:
                    # If ID format is unexpected, search all dates
                    recordings_dir = self.data_root / "recordings"
                    for date_dir in recordings_dir.iterdir():
                        if date_dir.is_dir():
                            wav_file = date_dir / f"{id}.wav"
                            if wav_file.exists():
                                date_formatted = date_dir.name
                                break
                    else:
                        raise HTTPException(status_code=404, detail=f"Transmission {id} not found")
                
                # Look for files
                recordings_dir = self.data_root / "recordings" / date_formatted
                wav_file = recordings_dir / f"{id}.wav"
                meta_file = wav_file.with_suffix(".meta.json")
                transcript_file = wav_file.with_suffix(".transcript.json")
                
                if not wav_file.exists() or not meta_file.exists():
                    raise HTTPException(status_code=404, detail=f"Transmission {id} not found")
                
                # Load metadata
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                
                # Create transmission object
                transmission = {
                    "id": id,
                    "date": date_formatted,
                    "timestamp_utc": metadata.get("timestamp_utc", ""),
                    "audio_path": str(wav_file.relative_to(self.data_root)),
                    "metadata_path": str(meta_file.relative_to(self.data_root)),
                    "duration_sec": metadata.get("duration_sec", 0),
                }
                
                # Add transcription if available
                if transcript_file.exists():
                    transmission["transcription_path"] = str(transcript_file.relative_to(self.data_root))
                    try:
                        with open(transcript_file, "r") as tf:
                            transmission["transcription"] = json.load(tf)
                    except Exception as e:
                        logger.error(f"Error loading transcription: {e}")
                
                return transmission
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting transmission: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/dates")
        async def get_dates():
            """Get list of available dates from recordings directory."""
            try:
                recordings_dir = self.data_root / "recordings"
                dates = []
                
                if recordings_dir.exists() and recordings_dir.is_dir():
                    for date_dir in recordings_dir.iterdir():
                        if date_dir.is_dir() and any(date_dir.glob("*.wav")):
                            dates.append(date_dir.name)
                
                # Sort dates (newest first)
                dates.sort(reverse=True)
                
                return {"dates": dates}
            except Exception as e:
                logger.error(f"Error getting dates: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/media/{path:path}")
        async def get_media(path: str):
            """Serve media files."""
            try:
                file_path = self.data_root / path
                
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail=f"File {path} not found")
                
                return FileResponse(file_path)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error serving media: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok"}
    
    def start(self):
        """Start the web server in a background thread."""
        if self.thread:
            return
        
        def run_server():
            """Run the server."""
            try:
                uvicorn.run(
                    self.app,
                    host=self.web_config.bind_host,
                    port=self.web_config.port,
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"Error running web server: {e}")
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        logger.info(f"Web server started at http://{self.web_config.bind_host}:{self.web_config.port}")
    
    def stop(self):
        """Stop the web server."""
        # Note: There's no clean way to stop a running uvicorn server in a thread
        # In a real application, you might want to use a more sophisticated approach
        logger.info("Web server stopping (note: may not stop cleanly)")
        self.thread = None
