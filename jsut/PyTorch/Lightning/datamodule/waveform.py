from typing import Callable, List, Optional, Union

# currently there is no stub in pytorch lightning
import pytorch_lightning as pl  # type: ignore
from torch.tensor import Tensor
from torch.utils.data import random_split, DataLoader

from ...dataset.waveform import NpVCC2016_wave, Speaker


class NpVCC2016_wave_DataModule(pl.LightningDataModule):
    """
    npVCC2016 speech corpus's PyTorch Lightning datamodule
    """

    def __init__(
        self,
        batch_size: int,
        download: bool,
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = lambda i: i,
        corpus_adress: Optional[str] = None,
        dataset_adress: Optional[str] = None,
        resample_sr: Optional[int] = None,
    ):
        super().__init__()
        self.n_batch = batch_size
        self.download = download
        self.speakers = speakers
        self.transform = transform
        self.corpus_adress = corpus_adress
        self.dataset_adress = dataset_adress
        self._resample_sr = resample_sr

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            dataset_train = NpVCC2016_wave(
                train=True,
                speakers=self.speakers,
                download_corpus=self.download,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
                transform=self.transform
            )
            n_train = len(dataset_train)
            self.data_train, self.data_val = random_split(
                dataset_train, [n_train - 10, 10]
            )
        if stage == "test" or stage is None:
            self.data_test = NpVCC2016_wave(
                train=False,
                speakers=self.speakers,
                download_corpus=self.download,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
                transform=self.transform
            )

    def train_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_train, batch_size=self.n_batch)

    def val_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_val, batch_size=self.n_batch)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.data_test, batch_size=self.n_batch)


if __name__ == "__main__":
    print("This is datamodule/waveform.py")
    # If you use batch (n>1), transform function for Tensor shape rearrangement is needed
    dm_npVCC_wave = NpVCC2016_wave_DataModule(1, download=True)

    # download & preprocessing
    dm_npVCC_wave.prepare_data()

    # runtime setup
    dm_npVCC_wave.setup(stage="fit")

    # yield dataloader
    dl = dm_npVCC_wave.train_dataloader()
    print(next(iter(dl)))
    print("datamodule/waveform.py test passed")