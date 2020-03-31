from torch.autograd import Variable
from .AudioCommon import *
from .DataBlock import *

import matplotlib.pyplot as plt
import torch
from fastai import *
from fastai.text import *
from fastai.vision import *
from scipy.ndimage.interpolation import shift
import torch
import librosa
import torchaudio
from torchaudio import transforms


def show_in_out(s: AudioData, r: AudioData):
    """Helper to plot input and output signal in different colors"""
    if s is not None:
        plt.plot(s.sig, 'm', label="Orig.")
    if r is not None:
        plt.plot(r.sig, 'c', alpha=0.5, label="Transf.")
    plt.legend()
    plt.show()


def tfm_tester(testSignal: AudioData, tfm, **kwargs):
    # TODO: track performance of execution
    tfms = listify(tfm)
    ret = testSignal  # default pass
    for t in tfms:
        ret = t(ret, **kwargs)
    print("↓ Original ↓")
    display(AudioItem(testSignal))
    print("↓ Transformed ↓")
    display(AudioItem(ret))
    show_in_out(testSignal, ret)


def tfm_shift(ad: AudioData, max_pct=0.6):
    v = (.5 - random.random())*max_pct*len(ad.sig)
    sig = shift(ad.sig, v, cval=.0)
    sig = torch.tensor(sig)
    return AudioData(sig=sig, sr=ad.sr)


def tfm_add_white_noise(ad: AudioData, noise_scl=0.005, **kwargs) -> AudioData:
    noise = torch.randn_like(ad.sig) * noise_scl
    return AudioData(ad.sig + noise, ad.sr)


def tfm_modulate_volume(ad: AudioData, lower_gain=.1, upper_gain=1.2, **kwargs) -> AudioData:
    modulation = random.uniform(lower_gain, upper_gain)
    return AudioData(ad.sig * modulation, ad.sr)


def tfm_random_cutout(ad: AudioData, pct_to_cut=.15, **kwargs) -> AudioData:
    """Randomly replaces `pct_to_cut` of signal with silence. Similar to grainy radio."""
    copy = ad.sig.clone()
    mask = (torch.rand_like(copy) > pct_to_cut).float()
    masked = copy * mask
    return AudioData(masked, ad.sr)


def tfm_pad_with_silence(ad: AudioData, pct_to_pad=.15, min_to_pad=None, max_to_pad=None, **kwargs) -> AudioData:
    """Adds silence to beginning or end of signal, simulating microphone cut at start of end of audio."""
    if max_to_pad is None:
        max_to_pad = int(ad.sig.shape[0] * 0.15)
    if min_to_pad is None:
        min_to_pad = -max_to_pad
    pad = random.randint(min_to_pad, max_to_pad)
    copy = ad.sig.clone()
    if pad >= 0:
        copy[0:pad] = 0
    else:
        copy[pad:] = 0
    return AudioData(copy, ad.sr)


def tfm_pitch_warp(ad: AudioData, shift_by_pitch=None, bins_per_octave=12, **kwargs) -> AudioData:
    """CAUTION - slow!"""
    min_len = 600  # librosa requires a signal of length at least 500
    copy = ad.sig.clone()
    if (copy.shape[0] < min_len):
        copy = torch.cat((copy, torch.zeros(min_len - copy.shape[0])))
    if shift_by_pitch is None:
        shift_by_pitch = random.uniform(-3, 3)
    sig = torch.tensor(librosa.effects.pitch_shift(
        np.array(copy), ad.sr, shift_by_pitch, bins_per_octave))
    return AudioData(sig, ad.sr)


def tfm_down_and_up(ad: AudioData, sr_divisor=2, **kwargs) -> AudioData:
    """CAUTION - slow!"""
    copy = np.array(ad.sig.clone())
    down = librosa.audio.resample(copy, ad.sr, ad.sr/sr_divisor)
    sig = torch.tensor(librosa.audio.resample(down, ad.sr/sr_divisor, ad.sr))
    return AudioData(sig, ad.sr)


def tfm_signal_mixup(signal, **kwargs):
    pass


def tfm_pad_to_max(ad: AudioData, mx=1000):
    """Pad tensor with zeros (silence) until it reaches length `mx`"""
    copy = ad.sig.clone()
    padded = torchaudio.transforms.PadTrim(max_len=mx)(copy[None, :]).squeeze()
    return AudioData(padded, ad.sr)


def tfm_pad_or_trim(ad: AudioData, mx, trim_section="mid", pad_at_end=True, **kwargs):
    """Pad tensor with zeros (silence) until it reaches length `mx` frames, or trim clip to length `mx` frames"""
    sig = ad.sig.clone()
    siglen = len(sig)
    if siglen < mx:
        diff = mx - siglen
        # Maintain input tensor device & type params
        padding = sig.new_zeros(diff)
        nsig = torch.cat((sig, padding)) if pad_at_end else torch.cat(
            (padding, sig))
    else:
        if trim_section not in {"start", "mid", "end"}:
            raise ValueError(
                f"'trim_section' argument must be one of 'start', 'mid' or 'end', got '{trim_section}'")
        if trim_section == "mid":
            nsig = sig.narrow(0, (siglen // 2) - (mx // 2), mx)
        elif trim_section == "end":
            nsig = sig.narrow(0, siglen-mx, mx)
        else:
            nsig = sig.narrow(0, 0, mx)
    return AudioData(sig=nsig, sr=ad.sr)


def tfm_log(x, show=False, logOnlyFirst=True, msg=''):
    '''Fake transformation that logs x shape'''
    # TODO: implements the optional "show" and "logOnlyFirst"
    if isinstance(x, AudioData):
        print(f'{msg}{type(x).__name__} >> Shape of signal: {x.sig.shape}  sr: {x.sr}')
    elif hasattr(x, 'shape'):
        print(f'{msg}{type(x).__name__} >> Shape: {x.shape}')
    else:
        print(f'{msg}{type(x).__name__}')
    return x


def _check_is_variable(tensor):
    if isinstance(tensor, torch.Tensor):
        is_variable = False
        tensor = Variable(tensor, requires_grad=False)
    elif isinstance(tensor, Variable):
        is_variable = True
    else:
        raise TypeError(
            "tensor should be a Variable or Tensor, but is {}".format(type(tensor)))

    return tensor, is_variable


class SPEC2DB(object):
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.

    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is -80.
    """

    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        self.top_db = -top_db if top_db > 0 else top_db
        self.multiplier = 10. if stype == "power" else 20.

    def __call__(self, spec):

        spec, is_variable = _check_is_variable(spec)
        spec_db = self.multiplier * \
            torch.log10(spec / spec.max())  # power -> dB
        if self.top_db is not None:
            spec_db = torch.max(spec_db, spec_db.new([self.top_db]))
        return spec_db if is_variable else spec_db.data


def tfm_extract_signal(ad: AudioData, **kwargs):
    return ad.sig


def tfm_spectro(ad: AudioData, sr=16000, to_db_scale=False, n_fft=400,
                ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128, **kwargs):
    # We must reshape signal for torchaudio to generate the spectrogram.
    # Note we don't pass **kwargs to stop interference from other params in the tfms.
    mel = transforms.MelSpectrogram(sr=ad.sr, n_mels=n_mels, n_fft=n_fft, ws=ws, hop=hop,
                                    f_min=f_min, f_max=f_max, pad=pad,)(ad.sig.reshape(1, -1))
    mel = mel.permute(0, 2, 1)  # swap dimension...
    if to_db_scale:
        mel = SPEC2DB(stype='magnitude', top_db=f_max)(mel)
    mel.sr = ad.sr
    mel.sig = ad.sig
    return mel


def tfm_spectro_stft(ad: AudioData, n_mels=128, n_fft=480, hop_length=160, win_length=480, window='hamming', **kwargs):
    # https://www.kaggle.com/haqishen/augmentation-methods-for-audio
    # https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89
    stft = librosa.stft(np.array(ad.sig), n_fft=n_fft, hop_length=hop_length)
    stft_magnitude, stft_phase = librosa.magphase(stft)
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
    ret = torch.tensor(stft_magnitude_db)
    ret = ret.unsqueeze(0)
    ret.sr = ad.sr
    ret.sig = ad.sig
    return ret


def get_audio_transforms(spectro: bool = False,
                         white_noise: bool = True,
                         shift_max_pct: float = .6,
                         modulate_volume: bool = True,
                         random_cutout: bool = True,
                         pad_with_silence: bool = True,
                         pitch_warp: bool = True,
                         down_and_up: bool = True,
                         mx_to_pad: int = 1000,
                         xtra_tfms: Optional[Collection[Transform]] = None,
                         **kwargs) -> Collection[Transform]:
    "Utility func to easily create a list of audio transforms."
    res = []
    if shift_max_pct:
        res.append(partial(tfm_shift, max_pct=shift_max_pct))
    if white_noise:
        res.append(partial(tfm_add_white_noise, **kwargs))
    if modulate_volume:
        res.append(partial(tfm_modulate_volume, **kwargs))
    if random_cutout:
        res.append(partial(tfm_random_cutout, **kwargs))
    if pad_with_silence:
        res.append(partial(tfm_pad_with_silence, **kwargs))
    if pitch_warp:
        res.append(partial(tfm_pitch_warp, **kwargs))
    if down_and_up:
        res.append(partial(tfm_down_and_up, **kwargs))
    res.append(partial(tfm_pad_to_max, mx=mx_to_pad))
    final_transform = tfm_extract_signal
    if spectro:
        final_transform = partial(tfm_spectro, **kwargs)
    res.append(final_transform)
    #       train                   , valid
    return (res + listify(xtra_tfms), [partial(tfm_pad_to_max, mx=mx_to_pad), final_transform])
