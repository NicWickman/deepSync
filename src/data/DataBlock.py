from AudioCommon import AudioData
from fastai.utils import *
from fastai.vision import *
from IPython.display import Audio
import torchaudio
from torchaudio import transforms


class AudioItem(ItemBase):
    def __init__(self, data: AudioData, **kwargs):
        self.data = data  # Always flatten out to single dimension signal!
        self.kwargs = kwargs

    def __str__(self):
        if isinstance(self.data, AudioData):
            return f'{self.__class__.__name__}: {self.duration}sec ({len(self)} @ {self.data.sr}hz).'
        else:
            return f'{type(self.data)}: {self.data.shape}'

    def __len__(self): return self.data.sig.shape[0]
    def _repr_html_(
        self): return f'{self.__str__()}<br />{self.ipy_audio._repr_html_()}'

    def show(self, title: Optional[str] = None, **kwargs):
        "Show sound on `ax` with `title`, using `cmap` if single-channel, overlaid with optional `y`"
        self.hear(title=title)
        print(self.data.shape)
        if len(self.data.shape) > 2:
            display(Image(self.data))

    def hear(self, title=None):
        if title is not None:
            print(title)
        display(self.ipy_audio)

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = tfm(self.data)
        return self

    @property
    def shape(self):
        return self.data.sig.shape

    @property
    def ipy_audio(self):
        return Audio(data=self.data.sig, rate=self.data.sr)

    @property
    def duration(self): return len(self.data.sig)/self.data.sr


class AudioDataBunch(DataBunch):
    def hear_ex(self, rows: int = 3, ds_type: DatasetType = DatasetType.Valid, **kwargs):
        batch = self.dl(ds_type).dataset[:rows]
        self.train_ds.hear_xys(batch.x, batch.y, **kwargs)

    def batch_stats(self, funcs: Collection[Callable] = None, ds_type: DatasetType = DatasetType.Train) -> Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean, torch.std])
        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()
        # raw signal has no channels so we unsqueeze 1st axis to add one to be flattened by channel_view
        if(len(x.shape) == 2):
            x = x.unsqueeze(1)
        return [func(channel_view(x), 1) for func in funcs]

    def normalize(self, stats: Collection[Tensor] = None, do_x: bool = True, do_y: bool = False) -> None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self, 'norm', False):
            raise Exception('Can not call normalize twice')
        if stats is None:
            self.stats = self.batch_stats()
        else:
            self.stats = stats
        self.norm, self.denorm = normalize_funcs(
            *self.stats, do_x=do_x, do_y=do_y)
        self.add_tfm(self.norm)
        return self


class AudioList(ItemList):
    _bunch = AudioDataBunch

    # TODO: __REPR__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, i):
        item = self.items[i]
        if isinstance(item, (Path, str)):
            return AudioItem(AudioData.load(str(item)))
        if isinstance(item, (tuple, np.ndarray)):  # data,sr
            return AudioItem(AudioData(item[0], item[1]))
        print('Format not supported!', file=sys.stderr)
        raise

    def reconstruct(self, t: Tensor): return Image(
        t.transpose(1, 2))  # FIXME!! No Image here

    def hear_xys(self, xs, ys, **kwargs):
        for x, y in zip(xs, ys):
            x.hear(title=y, **kwargs)

    # TODO: example with from_folder
    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, **kwargs) -> ItemList:
        extensions = ifnone(extensions, AUDIO_EXTENSIONS)
        return super().from_folder(path=path, extensions=extensions, **kwargs)
