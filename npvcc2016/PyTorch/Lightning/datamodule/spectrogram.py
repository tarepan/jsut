from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
from torch.tensor import Tensor
from torch.utils.data import random_split, DataLoader

from ...dataset.waveform import Speaker
from ...dataset.spectrogram import NpVCC2016_spec


class NpVCC2016_spec_DataModule(pl.LightningDataModule):
    """
    npVCC2016_spec dataset's PyTorch Lightning datamodule
    """

    def __init__(
        self,
        batch_size: int,
        download: bool,
        dir_root: str = "./data/",
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = lambda i: i,
        corpus_adress: Optional[str] = None,
        dataset_adress: str = "./data/datasets/npVCC2016_spec/archive/dataset.zip",
        zipfs: bool = False,
    ):
        super().__init__()
        self.n_batch = batch_size
        self.download = download
        self.dir_root = dir_root
        self.speakers = speakers
        self.transform = transform
        self.corpus_adress = corpus_adress
        self.dataset_adress = dataset_adress
        self.zipfs = zipfs
        # transforms.Compose([transforms.ToTensor()])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            dataset_train = NpVCC2016_spec(
                train=True,
                speakers=self.speakers,
                transform=self.transform,
                download_corpus=self.download,
                dir_data=self.dir_root,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
                zipfs=self.zipfs,
                compression=False,
                cache=False
            )
            n_train = len(dataset_train)
            self.data_train, self.data_val = random_split(
                dataset_train, [n_train - 10, 10]
            )
        if stage == "test" or stage is None:
            self.data_test = NpVCC2016_spec(
                train=False,
                speakers=self.speakers,
                transform=self.transform,
                download_corpus=self.download,
                dir_data=self.dir_root,
                corpus_adress=self.corpus_adress,
                dataset_adress=self.dataset_adress,
                zipfs=self.zipfs,
                compression=False,
                cache=False
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
    dm_npVCC_spec = NpVCC2016_spec_DataModule(1, download=True)

    # download & preprocessing
    dm_npVCC_spec.prepare_data()

    # runtime setup
    dm_npVCC_spec.setup(stage="fit")

    # yield dataloader
    dl = dm_npVCC_spec.train_dataloader()
    print(next(iter(dl)))
    print("datamodule/waveform.py test passed")