"""Shared model cache.

Each SmileBERTa model is ~176 MB of float32 weights (~250 MB resident once loaded).
Render's free tier has 512 MB of RAM, so by default only one model stays in memory
(MODEL_CACHE_SIZE=1) and the least-recently-used one is evicted when the other is
needed. Set MODEL_CACHE_SIZE=2 on machines with more memory to keep both warm.
Outputs are identical either way — this only trades a few seconds of reload time.
"""
import gc
import os
import threading
from collections import OrderedDict

from transformers import RobertaForMaskedLM, RobertaTokenizer

CACHE_SIZE = max(1, int(os.environ.get('MODEL_CACHE_SIZE', '1')))
_cache = OrderedDict()
_lock = threading.RLock()


def get_model(name):
    """Return (model, tokenizer), loading it if needed. Callers hold the lock during inference."""
    with _lock:
        if name in _cache:
            _cache.move_to_end(name)
            return _cache[name]
        while len(_cache) >= CACHE_SIZE:
            _cache.popitem(last=False)
            gc.collect()
        tokenizer = RobertaTokenizer.from_pretrained(name)
        model = RobertaForMaskedLM.from_pretrained(name)
        model.eval()
        _cache[name] = (model, tokenizer)
        return _cache[name]


def model_lock():
    return _lock


def loaded():
    with _lock:
        return list(_cache.keys())


def unload_all():
    """Free every cached model and hand the memory back to the OS (used before docking)."""
    with _lock:
        _cache.clear()
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass
