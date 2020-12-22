"""
# Corpus/Dataset guide
Corpus: Distributed data
Dataset: Processed corpus for specific purpose
For example, JSUT corpus contains waves, and can be processed into JSUT-spec dataset which is made of spectrograms.
"""

# from typing import Callable, List, Literal, NamedTuple # >= Python3.8
from typing import Callable, List, NamedTuple, Optional
from pathlib import Path

from torch import Tensor, save, load
from torch.utils.data import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Resample

from ...fs import hash_args, try_to_acquire_archive_contents, save_archive
from ...corpus import ItemIdNpVCC2016, Mode, NpVCC2016, Speaker


def get_dataset_wave_path(dir_dataset: Path, id: ItemIdNpVCC2016) -> Path:
    return dir_dataset / id.mode / id.speaker / "waves" / f"{id.serial_num}.wave.pt"


def preprocess_as_wave(corpus: NpVCC2016, dir_dataset: Path, new_sr: Optional[int] = None) -> None:
    """
    Transform npVCC2016 corpus contents into waveform Tensor.
    Before this preprocessing, corpus contents should be deployed.

    Args:
        new_sr: If specified, resample with specified sampling rate.
    """
    for id in corpus.get_identities():
        waveform, _sr_orig = load_wav(corpus.get_item_path(id))
        if new_sr is not None:
            waveform = Resample(_sr_orig, new_sr)(waveform)
        # :: [1, Length] -> [Length,]
        waveform: Tensor = waveform[0, :]
        path_wave = get_dataset_wave_path(dir_dataset, id)
        path_wave.parent.mkdir(parents=True, exist_ok=True)
        save(waveform, path_wave)


class Datum_NpVCC2016_wave(NamedTuple):
    """
    Datum of NpVCC2016 dataset
    """

    waveform: Tensor
    label: str


class NpVCC2016_wave(Dataset): # I failed to understand this error
    """
    Audio waveform dataset from npVCC2016 non-parallel speech corpus.
    This dataset yield (audio, label).
    """
    def __init__(
        self,
        train: bool,
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        download_corpus: bool = False,
        corpus_adress: Optional[str] = None,
        dataset_adress: Optional[str] = None,
        resample_sr: Optional[int] = None,
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            speakers: Selected speaker list.
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

        self._corpus = NpVCC2016(download_corpus, corpus_adress)
        dirname = hash_args(train, speakers, download_corpus, corpus_adress, dataset_adress, resample_sr)
        self._path_contents_local = Path(".")/"tmp"/"npVCC2016_wave"/"contents"/dirname
        dataset_adress = dataset_adress if dataset_adress else str(Path(".")/"tmp"/"npVCC2016_wave"/"archive"/f"{dirname}.zip")

        # Prepare data identities.
        mode: Mode = "trains" if train else "evals"
        self._ids: List[ItemIdNpVCC2016] = list(
            filter(lambda id: id.speaker in speakers,
                filter(lambda id: id.mode == mode,
                    self._corpus.get_identities()
        )))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(self._path_contents_local, dataset_adress, True)
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
        preprocess_as_wave(self._corpus, self._path_contents_local, self._resample_sr)

    def _load_datum(self, id: ItemIdNpVCC2016) -> Datum_NpVCC2016_wave:
        waveform: Tensor = load(get_dataset_wave_path(self._path_contents_local, id))
        return Datum_NpVCC2016_wave(self._transform(waveform), f"{id.mode}-{id.speaker}-{id.serial_num}")

    def __getitem__(self, n: int) -> Datum_NpVCC2016_wave:
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
    NpVCC2016_wave(train=True, download_corpus=True)

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