from setuptools import setup, find_packages

setup(
    name="lauschomat",
    version="0.1.0",
    description="Realtime radio transcription system with squelch detection and NVIDIA Parakeet TDT",
    author="Lauschomat Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pydantic>=1.8.2,<2.0.0",  # NeMo has compatibility issues with pydantic 2.x
        "pyyaml>=6.0",
        "sounddevice>=0.4.3",
        "soundfile>=0.10.3",
        "webrtcvad>=2.0.10",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.5",
        "jinja2>=3.1.2",
        "aiofiles>=23.1.0",
    ],
    extras_require={
        "transcribe": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "librosa>=0.9.0",
            "nemo_toolkit[asr]>=1.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lauschomat-capture=lauschomat.capture.main:main",
            "lauschomat-transcribe=lauschomat.transcribe.main:main",
            "lauschomat-web=lauschomat.web.main:main",
            "lauschomat-dev=lauschomat.dev.main:main",
            "lauschomat-test-parakeet=tests.test_parakeet:main",
        ],
    },
)
