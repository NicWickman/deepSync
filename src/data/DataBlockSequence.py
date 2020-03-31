from .AudioCommon import *
from .DataBlock import *
from .DataAugmentation import *

import mimetypes
from fastai.vision import *
from fastai.text import *
import torchaudio
from torchaudio import transforms

# for jupyter Display
from IPython.display import Audio


class AudioSequenceItem(ItemBase):
    def __init__(self,
                 data: AudioData,
                 sample_len=401,
                 stride_len=200,
                 max_seqs=20,
                 **kwargs):
        # chopping up one signal item to [0,1,2], [1,2,3], [2,3,4]...
        chopped = []
        numOfChunks = ((data.sig.shape[0]-sample_len)//stride_len)+1

        for i in range(0, numOfChunks*stride_len, stride_len):
            if (len(chopped) >= max_seqs):
                break
            chop = data.sig[i:i+sample_len].clone()
            chopped.append(chop)

        self.data = (chopped, data.sr)
        self.kwargs = kwargs

    @property
    def sr(self):
        return self.data[1]

    @property
    def seq(self):
        return self.data[0]

    @property
    def num_seqs(self):
        return len(self.seq)

    def __str__(self):
        return f'Length: {len(self.seq)} | Shape: {self.seq[0].shape} | Sample Rate: {self.sr}'

    @property
    def size(self):
        return self.seq[0].size()

    def apply_tfms(self, tfms):
        modified = self.data
        for tfm in tfms:
            modified = tfm(modified)
        return modified

    @property
    def shape(self):
        return self.seq[0].shape

    @classmethod
    def from_file(cls, fp: Path, **kwargs):
        return AudioSequenceItem(AudioData.load(fp), **kwargs)


def _maybe_squeeze(arr): return (arr if is1d(arr) else np.squeeze(arr))


class AudioSequenceList(ItemList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_df(cls, df: DataFrame, path: PathOrStr = '.', **kwargs) -> 'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        inputs = df.iloc[:, ]  # FIX ME
        assert inputs.isna().sum().sum(
        ) == 0, f"You have NaN values in column(s) of your dataframe, please fix it."
        res = cls(items=_maybe_squeeze(inputs.values),
                  path=path, inner_df=df, **kwargs)
        return res


class AudioSequenceDataBunch(DataBunch):
    @classmethod
    def from_df(cls,
                path: PathOrStr,
                train_df: DataFrame,
                valid_df: DataFrame,
                audio_cols=[],
                **kwargs) -> DataBunch:
        src = ItemLists(path, AudioSequenceList.from_df(
            train_df, path), AudioSequenceList.from_df(valid_df, path))
        # TODO: toggle classifier or LM here (toggle labels)
        labeled = src.label_const(0)
        def extract_seq(x): return x[0]
        tfms = [[extract_seq], [extract_seq]]
        labeled.transform(tfms)
        return labeled.databunch(**kwargs)
