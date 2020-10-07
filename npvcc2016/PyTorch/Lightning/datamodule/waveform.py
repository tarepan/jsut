from typing import Callable, List, Union

# currently there is no stub in pytorch lightning
import pytorch_lightning as pl  # type: ignore
from torch.tensor import Tensor
from torch.utils.data import random_split, DataLoader

## start parent module import
import sys
from pathlib import Path

dir_PyTorch = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(dir_PyTorch))
from ...dataset.waveform import NpVCC2016, Speaker  # type: ignore

## end


class NpVCC2016DataModule(pl.LightningDataModule):
    """
    npVCC2016 speech corpus's PyTorch Lightning datamodule
    """

    def __init__(
        self,
        batch_size: int,
        download: bool,
        dir_root: str = ".",
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = lambda i: i,
    ):
        super().__init__()
        self.n_batch = batch_size
        self.download = download
        self.dir_root = dir_root
        self.speakers = speakers
        self.transform = transform
        # transforms.Compose([transforms.ToTensor()])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self, *args, **kwargs) -> None:
        NpVCC2016(self.dir_root, train=True, download=self.download)

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            dataset_train = NpVCC2016(
                self.dir_root, train=True, transform=self.transform
            )
            n_train = len(dataset_train)
            self.data_train, self.data_val = random_split(
                dataset_train, [n_train - 10, 10]
            )
        if stage == "test" or stage is None:
            self.data_test = NpVCC2016(
                self.dir_root, train=False, transform=self.transform
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
    dm_npVCC_wave = NpVCC2016DataModule(1, download=True)

    # download & preprocessing
    dm_npVCC_wave.prepare_data()

    # runtime setup
    dm_npVCC_wave.setup(stage="fit")

    # yield dataloader
    dl = dm_npVCC_wave.train_dataloader()
    print(next(iter(dl)))
    print("datamodule/waveform.py test passed")