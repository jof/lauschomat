#!/usr/bin/env python3
"""
Script to list available NeMo ASR models.
"""
import sys
from pathlib import Path

# Add parent directory to path to import from lauschomat
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nemo.collections.asr as nemo_asr
    
    print("Available NeMo ASR models:")
    print("\nPretrained Encoder-Decoder CTC Models:")
    for model_name in nemo_asr.models.EncDecCTCModel.list_available_models():
        print(f"  - {model_name}")
    
    print("\nPretrained Encoder-Decoder RNNT Models:")
    for model_name in nemo_asr.models.EncDecRNNTBPEModel.list_available_models():
        print(f"  - {model_name}")
        
    print("\nPretrained Conformer-CTC Models:")
    for model_name in nemo_asr.models.EncDecCTCModelBPE.list_available_models():
        print(f"  - {model_name}")
    
    # Try to list other model types
    print("\nOther available model classes:")
    for cls_name in dir(nemo_asr.models):
        if cls_name.startswith('EncDec') and hasattr(nemo_asr.models, cls_name):
            cls = getattr(nemo_asr.models, cls_name)
            if hasattr(cls, 'list_available_models'):
                try:
                    models = cls.list_available_models()
                    if models:
                        print(f"\n{cls_name}:")
                        for model_name in models:
                            print(f"  - {model_name}")
                except Exception as e:
                    pass
    
except ImportError:
    print("NeMo ASR module not available. Install with: pip install nemo_toolkit[asr]")
    sys.exit(1)
