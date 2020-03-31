from pathlib import Path
import mimetypes
import torchaudio


class AudioData:
    '''Holds basic information from audio signal'''

    def __init__(self, sig, sr=48000):
        self.sig = sig.reshape(-1)  # We want single dimension data
        self.sr = sr

    @classmethod
    def load(cls, fileName, **kwargs):
        p = Path(fileName)
        if p.exists() & str(p).lower().endswith(AUDIO_EXTENSIONS):
            signal, samplerate = torchaudio.load(str(fileName))
            return AudioData(signal, samplerate)
        raise Exception(
            f"Error while processing {fileName}: file not found or does not have valid extension: {AUDIO_EXTENSIONS}")
