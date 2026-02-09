import torchaudio
import warnings

# Filter user warnings
warnings.simplefilter('ignore', UserWarning)

print("Torchaudio version:", torchaudio.__version__)
try:
    print("Available backends:", torchaudio.list_audio_backends())
except AttributeError:
    print("list_audio_backends not found (likely >=2.10)")

try:
    torchaudio.set_audio_backend("soundfile")
    print("Set backend to soundfile successfully (via set_audio_backend)")
except AttributeError:
    print("set_audio_backend not found")
except Exception as e:
    print("Failed to set backend:", e)

try:
    # Try alternate location in newer versions
    from torchaudio.backend import set_audio_backend
    set_audio_backend("soundfile")
    print("Set backend to soundfile successfully (via backend.set_audio_backend)")
except ImportError:
    pass
except Exception as e:
    print("Failed via backend module:", e)
