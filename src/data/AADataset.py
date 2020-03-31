from pathlib import Path
from ast import literal_eval
from torch.utils.data import Dataset
import os
import pandas as pd
import torchaudio
import tarfile
from torchaudio.datasets.utils import walk_files


class AADataset(Dataset):
    """
    Create a Dataset for Audio. Each item is a tuple of the form:
    (waveform, sample_rate, labels)
    """

    def __init__(
        self,
        root,
        data_path=Path('data/raw'),
        anim_path=Path('data.csv'),
        audio_path=Path('audio'),
        audio_ext='.aif',
        unpack_audio=False,
        transform=None,
        target_transform=None,
    ):

        self.transform = transform
        self.target_transform = target_transform

        self._audio_rar = root/data_path/Path('audio.tar.gz')
        self._audio_path = root/data_path/audio_path
        self._anim_path = root/data_path/anim_path
        self._ext_audio = audio_ext

        if unpack_audio == True:
            if not os.path.exists(self._audio_path):
                os.mkdir(self._audio_path)
            print(self._anim_path)
            tarfile.open(self._audio_rar, 'r:gz').extractall(self._audio_path)

        if not os.path.isdir(self._audio_path):
            raise RuntimeError(
                "Dataset not found."
            )

        self.df = self._process_df(pd.read_csv(self._anim_path))

        walker = walk_files(
            self._audio_path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )

        # self._walker = list(walker)
        self._walker = [x for x in list(walker) if len(
            self.df[self.df['Audio File'].str.contains(x)]['jawTrans_ty']) > 0]  # Filter out missing audio

    def _process_df(self, df):
        def convert_to_list(x):
            try:
                y = literal_eval(x)
                return y
            except:
                return x

        def convert_floats(x):
            if isinstance(x, float):
                return [x]
            else:
                return x

        df['jawTrans_ty'] = df['jawTrans_ty'].fillna('0.0')
        df['jawTrans_ty'] = df['jawTrans_ty'].apply(
            lambda x: convert_to_list(x))
        df['jawTrans_ty'] = df['jawTrans_ty'].apply(
            lambda x: convert_floats(x))
        return df

    def _get_labels(self, fileid):
        y_vals = self.df[self.df['Audio File'].str.contains(
            fileid)]['jawTrans_ty'].values[0]
        return y_vals

    def _load_audio_item(self, fileid):
        # Read label
        labels = self._get_labels(fileid)

        # Read audio
        filepath = str(self._audio_path / Path(fileid)) + self._ext_audio

        waveform, sample_rate = torchaudio.load(filepath)

        return waveform, sample_rate, labels

    def __getitem__(self, n):
        fileid = self._walker[n]
        item = self._load_audio_item(fileid)

        waveform, sample_rate, labels = item

        labels = torch.Tensor(labels).unsqueeze(dim=0)

        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return waveform, labels

    def __len__(self):
        return len(self._walker)
