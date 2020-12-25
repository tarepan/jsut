from typing import Callable, List, NamedTuple, Optional
from pathlib import Path

from torch import Tensor, save, load
from torch.utils.data import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Resample
from corpuspy.components.archive import hash_args, save_archive, try_to_acquire_archive_contents

from ...corpus import ItemIdJSUT, Subtype, JSUT


def get_dataset_wave_path(dir_dataset: Path, id: ItemIdJSUT) -> Path:
    return dir_dataset / id.subtype / "waves" / f"{id.serial_num}.wave.pt"


def preprocess_as_wave(path_wav: Path, id: ItemIdJSUT, dir_dataset: Path, new_sr: Optional[int] = None) -> None:
    """Transform JSUT corpus contents into waveform Tensor.

    Before this preprocessing, corpus contents should be deployed.

    Args:
        path_wav: processded .wav path.
        id: Target item identity.
        dir_dataset: Dataset root path.
        new_sr: If specified, resample with specified sampling rate.
    """

    waveform, _sr_orig = load_wav(path_wav)
    if new_sr is not None:
        waveform = Resample(_sr_orig, new_sr)(waveform)
    # :: [1, Length] -> [Length,]
    waveform: Tensor = waveform[0, :]
    path_wave = get_dataset_wave_path(dir_dataset, id)
    path_wave.parent.mkdir(parents=True, exist_ok=True)
    save(waveform, path_wave)


class Datum_JSUT_wave(NamedTuple):
    """
    Datum of NpVCC2016 dataset
    """

    waveform: Tensor
    label: str


class JSUT_wave(Dataset): # I failed to understand this error
    """Audio waveform dataset from JSUT speech corpus.
    """

    def __init__(
        self,
        train: bool,
        subtypes: List[Subtype] = ["basic5000"],
        download_corpus: bool = False,
        corpus_adress: Optional[str] = None,
        dataset_adress: Optional[str] = None,
        resample_sr: Optional[int] = None,
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            subtypes: Corpus item subtypes for the dataset.
            download_corpus: Whether download the corpus or not when dataset is not found.
            corpus_adress: URL/localPath of corpus archive (e.g. `s3::` can be used). None use default URL.
            dataset_adress: URL/localPath of dataset archive (e.g. `s3::` can be used). None use default local path.
            resample_sr: If specified, resample with specified sampling rate.
            transform: Tensor transform on load.
        """
        # Design Notes:
        #   Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #   `download` is common option in torchAudio datasets.

        # Store parameters.
        self._resample_sr = resample_sr
        self._transform = transform

        self._corpus = JSUT(corpus_adress, download_corpus)
        dirname = hash_args(train, subtypes, download_corpus, corpus_adress, dataset_adress, resample_sr)
        JSUT_wave_root = Path(".")/"tmp"/"JSUT_wave"
        self._path_contents_local = JSUT_wave_root/"contents"/dirname
        dataset_adress = dataset_adress if dataset_adress else str(JSUT_wave_root/"archive"/f"{dirname}.zip")

        # Prepare data identities.
        self._ids: List[ItemIdJSUT] = list(filter(lambda id: id.subtype in subtypes, self._corpus.get_identities()))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(dataset_adress, self._path_contents_local)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents_local, dataset_adress)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """
        Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()
        print("Preprocessing...")
        for id in self._ids:
            path_wav = self._corpus.get_item_path(id)
            preprocess_as_wave(path_wav, id, self._path_contents_local, self._resample_sr)
        print("Preprocessed.")

    def _load_datum(self, id: ItemIdJSUT) -> Datum_JSUT_wave:
        waveform: Tensor = load(get_dataset_wave_path(self._path_contents_local, id))
        return Datum_JSUT_wave(self._transform(waveform), f"{id.subtype}-{id.serial_num}")

    def __getitem__(self, n: int) -> Datum_JSUT_wave:
        """Load the n-th sample from the dataset.
        Args:
            n: The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    print("This is waveform.py")
    # dataset preparation
    JSUT_wave(train=True, download_corpus=True)

    # # setup
    # dataset_train_full = NpVCC2016(".", train=True, download=False)
    # dataset_train_SF1_SM1 = NpVCC2016(
    #     ".", train=True, download=False, speakers=["SF1", "SM1"]
    # )
    # dataset_train_SF1 = NpVCC2016(".", train=True, download=False, speakers=["SF1"])
    # dataset_test_SF1 = NpVCC2016(".", train=False, download=False, speakers=["SF1"])

    # # check
    # n_full = len(dataset_train_full)
    # assert (
    #     n_full == 81 * 4
    # ), f"dataset_train_full should contains {81*4} datums, but there are {n_full}"

    # n_SF1_SM1 = len(dataset_train_SF1_SM1)
    # assert (
    #     n_SF1_SM1 == 81 * 2
    # ), f"dataset_train_SF1_SM1 should contains {81*2} datums, but there are {n_SF1_SM1}"

    # n_SF1 = len(dataset_train_SF1)
    # assert (
    #     n_SF1 == 81 * 1
    # ), f"dataset_train_SF1 should contains {81*1} datums, but there are {n_SF1}"

    # n_t_SF1 = len(dataset_test_SF1)
    # assert (
    #     n_t_SF1 == 54 * 1
    # ), f"dataset_test_SF1 should contains {54*1} datums, but there are {n_t_SF1}"

    # print("waveform.py test passed")