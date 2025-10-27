"""
Configuration management for Lauschomat components.
"""
import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class AudioConfig(BaseModel):
    """Audio device and capture configuration."""
    backend: str = "pulse"
    device_name: str = "default"
    channels: int = 1
    sample_rate: int = 16000
    format: str = "S16_LE"
    gain_db: float = 0.0


class SquelchConfig(BaseModel):
    """Squelch detection configuration."""
    method: str = "energy_hysteresis"
    frame_ms: int = 20
    threshold_open: float = -35.0
    threshold_close: float = -45.0
    min_open_ms: int = 80
    hang_ms: int = 800
    max_transmission_ms: int = 180000


class RecordingConfig(BaseModel):
    """Audio recording configuration."""
    codec: str = "wav"
    container: str = "wav"
    include_pre_roll_ms: int = 200
    include_post_roll_ms: int = 300
    filename_template: str = "{date}/{timestamp}_{seq}.{ext}"


class TransferConfig(BaseModel):
    """File transfer configuration."""
    method: str = "remote_fs"
    target_host: str = "gpu-server.local"
    target_path: str = "/var/lib/lauschomat/incoming"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    transfer_interval_sec: int = 10


class TranscriptionConfig(BaseModel):
    """Transcription configuration."""
    enabled: bool = True
    engine: str = "nemo_parakeet_tdt"
    model_name: str = "nvidia/parakeet-ctc-1.1b"
    device: str = "cuda:0"
    batch_size: int = 1
    language: str = "en-US"
    diarization: bool = False


class StorageConfig(BaseModel):
    """Storage and indexing configuration."""
    index_format: str = "jsonl"
    partitioning: str = "by_day"


class WebConfig(BaseModel):
    """Web server configuration."""
    enabled: bool = True
    bind_host: str = "0.0.0.0"
    port: int = 8080
    base_path: str = "/"
    static_dir: str = Field(default_factory=lambda: str(Path("/var/lib/lauschomat/web/static")))
    template_dir: str = Field(default_factory=lambda: str(Path("/var/lib/lauschomat/web/templates")))


class AppConfig(BaseModel):
    """Application-wide configuration."""
    mode: str = "production"
    data_root: str = Field(default_factory=lambda: str(Path("/var/lib/lauschomat")))
    tmp_dir: str = Field(default_factory=lambda: str(Path("/var/tmp/lauschomat")))
    timezone: str = "UTC"
    node_id: str = "default-node"
    incoming_dir: Optional[str] = None
    processed_dir: Optional[str] = None

    @validator("data_root", "tmp_dir", "incoming_dir", "processed_dir")
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return str(Path(os.path.expanduser(v)).absolute())


class Config(BaseModel):
    """Complete configuration for Lauschomat."""
    app: AppConfig = Field(default_factory=AppConfig)
    audio: Optional[AudioConfig] = None
    squelch: Optional[SquelchConfig] = None
    recording: Optional[RecordingConfig] = None
    transfer: Optional[TransferConfig] = None
    transcription: Optional[TranscriptionConfig] = None
    storage: Optional[StorageConfig] = None
    web: Optional[WebConfig] = None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.parse_obj(config_dict)

    @classmethod
    def from_env(cls, env_var: str = "LAUSCHOMAT_CONFIG") -> "Config":
        """Load configuration from the environment variable."""
        config_path = os.environ.get(env_var)
        if not config_path:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls.from_yaml(config_path)

    def get_data_path(self, *paths: str) -> Path:
        """Get a path relative to the data root."""
        return Path(self.app.data_root).joinpath(*paths)

    def get_tmp_path(self, *paths: str) -> Path:
        """Get a path relative to the temp directory."""
        return Path(self.app.tmp_dir).joinpath(*paths)


def load_config(config_path: Optional[Union[str, Path]] = None, env_var: str = "LAUSCHOMAT_CONFIG") -> Config:
    """Load configuration from a file or environment variable."""
    if config_path:
        return Config.from_yaml(config_path)
    try:
        return Config.from_env(env_var)
    except ValueError:
        # Default to development config if nothing specified
        return Config(app=AppConfig(mode="development", data_root="./data", tmp_dir="./tmp"))
