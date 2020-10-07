"""
# Corpus/Dataset guide

Corpus: Distributed data
Dataset: Processed corpus for specific purpose
For example, JSUT corpus contains waves, and can be processed into JSUT-spec dataset which is made of spectrograms.

Corpus is acquired then processed into dataset.
`_prepare_corpus` implements corpus preparation (download/localLoad).
`_preprocess_corpus` convert prepared corpus into dataset (== preprocessing).
This preprocess occur only once after Corpus construction.

Basic dataset yield corpus contents as dataset.
If extend basic dataset class and override `_preprocess_corpus`, it enable extended dataset (e.g. waveform => spectrogram).
Alternatively, super().__init__ & additional preprocessing in extended class's __init__ work well.
"""


from typing import Callable, List, Literal, NamedTuple
from itertools import chain
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
import torchaudio  # type: ignore
from torchaudio.datasets.utils import download_url, extract_archive  # type: ignore


Speaker = Literal["SF1", "SM1", "TF2", "TM3"]


class Datum_identity(NamedTuple):
    mode: Literal["evals", "trains"]
    speaker: Speaker
    serial_num: str


def list_up_wav_names(dir_path: Path) -> List[str]:
    """
    List up all .wav file names in the specified directory (no deep/recursive search)
    """
    files = filter(lambda path: path.is_file(), dir_path.iterdir())
    wavs = filter(lambda file: file.suffix == ".wav", files)
    return list(map(lambda wav: wav.stem, wavs))


class Datum_NpVCC2016(NamedTuple):
    """
    Datum of NpVCC2016 dataset
    """

    waveform: Tensor
    label: str


class NpVCC2016(Dataset):  # I failed to understand this error
    """
    Audio waveform dataset from npVCC2016 non-parallel speech corpus.
    This dataset yield (audio, label).
    Args:
        root: Root directory where the dataset's top level directory is found.
        train: train ? train_dataset : test_dataset
        download: Whether to download the dataset if it is not found at root path (common in torchAudio datasets).
        speakers: selection of npVCC2016 speakers for usage
        transform: transform on load
    """

    ver: str = "1.0.0"
    corpus_name: str = f"npVCC2016-{ver}"
    url: str = f"https://github.com/tarepan/npVCC2016Corpus/releases/download/v{ver}/{corpus_name}.zip"
    speakers = ("SF1", "SM1", "TF2", "TM3")

    def __init__(
        self,
        root: str,  # root could come from parser (e.g. argparse), so root::pathlib.Path could decrease usability
        train: bool,
        download: bool = False,
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        """
        Prepare (download or access local) corpus, then transform them as dataset with `_process_corpus()`.
        In extended class, overriding `_preprocess_corpus` + calling `super().__init__()` enable extension.
        """
        # store parameters
        self._train = train
        self._trainEvals = "trains" if train else "evals"
        self._transform = transform

        # corpus/dataset preparation
        self._prepare_corpus(Path(root), train, download, speakers)
        self._preprocess_corpus()

    def _prepare_corpus(
        self, root: Path, train: bool, download: bool, speakers: List[Speaker]
    ):
        """
        Prepare (download or access local) the corpus
        """
        archive = root / f"{self.corpus_name}.zip"
        self._path_corpus = root / self.corpus_name

        # Download the corpus's archive from the url on root, if neither corpusDir nor corpusArchive exist
        if download & (not self._path_corpus.is_dir()) & (not archive.is_file()):
            download_url(self.url, str(root))
        # Extract corpus from archive, if corpusDir do not exists but corpusArchive exists
        if (not self._path_corpus.is_dir()) & (archive.is_file()):
            extract_archive(str(archive), str(self._path_corpus))
        # Check corpus directory existance
        if not self._path_corpus.is_dir():
            raise RuntimeError("Corpus directory not found. Check `download` param")

        # Extract corpus item identities
        ## directory strucutre: /("evals"|"trains")/(Speaker)/wavs/xxxxx.wav
        def speaker2dir(speaker: Speaker, trainEvals: Literal["trains", "evals"]):
            return self._path_corpus / trainEvals / speaker / "wavs"

        self._corpus_item_identities: List[Datum_identity] = []
        for corpusTE in ("trains", "evals"):
            self._corpus_item_identities.extend(
                chain.from_iterable(
                    map(
                        lambda speaker: map(
                            # type inference failure(`literal` as `str`), so no problem
                            lambda name: Datum_identity(corpusTE, speaker, name),  # type: ignore
                            list_up_wav_names(speaker2dir(speaker, corpusTE)),  # type: ignore
                        ),
                        self.speakers,
                    )
                )
            )
        # prepare datum identities
        dataset_trainEvals = "trains" if train else "evals"
        self._datum_identities = list(
            chain.from_iterable(
                map(
                    lambda speaker: map(
                        # type inference failure(`literal` as `str`), so no problem
                        lambda name: Datum_identity(dataset_trainEvals, speaker, name),  # type: ignore
                        list_up_wav_names(speaker2dir(speaker, dataset_trainEvals)),  # type: ignore
                    ),
                    speakers,
                )
            )
        )

    def _preprocess_corpus(self) -> None:
        """
        Transform corpus waveform/label into arbitrary data as datasets.
        In default, raw waveform and labels are yielded (== this function do nothing).
        """
        pass

    def _calc_path_wav(self, path_corpus: Path, id: Datum_identity) -> Path:
        return path_corpus / id.mode / id.speaker / "wavs" / f"{id.serial_num}.wav"

    def _load_datum(self, path_corpus: Path, id: Datum_identity) -> Datum_NpVCC2016:
        # no stub problem (see import parts) + torchaudio internal override (It is my guess. It looks like no-interface problem?)
        # pylint: disable=no-member
        waveform, _sr = torchaudio.load(self._calc_path_wav(path_corpus, id))  # type: ignore
        return Datum_NpVCC2016(self._transform(waveform), f"{id.mode}-{id.speaker}-{id.serial_num}")  # type: ignore

    def __getitem__(self, n: int) -> Datum_NpVCC2016:
        """Load the n-th sample from the dataset.
        Args:
            n: The index of the datum to be loaded
        """
        return self._load_datum(self._path_corpus, self._datum_identities[n])

    def __len__(self) -> int:
        return len(self._datum_identities)


if __name__ == "__main__":
    print("This is waveform.py")
    # dataset preparation
    # NpVCC2016(".", train=True, download=True)  # commented out for safety

    # setup
    dataset_train_full = NpVCC2016(".", train=True, download=False)
    dataset_train_SF1_SM1 = NpVCC2016(
        ".", train=True, download=False, speakers=["SF1", "SM1"]
    )
    dataset_train_SF1 = NpVCC2016(".", train=True, download=False, speakers=["SF1"])
    dataset_test_SF1 = NpVCC2016(".", train=False, download=False, speakers=["SF1"])

    # check
    n_full = len(dataset_train_full)
    assert (
        n_full == 81 * 4
    ), f"dataset_train_full should contains {81*4} datums, but there are {n_full}"

    n_SF1_SM1 = len(dataset_train_SF1_SM1)
    assert (
        n_SF1_SM1 == 81 * 2
    ), f"dataset_train_SF1_SM1 should contains {81*2} datums, but there are {n_SF1_SM1}"

    n_SF1 = len(dataset_train_SF1)
    assert (
        n_SF1 == 81 * 1
    ), f"dataset_train_SF1 should contains {81*1} datums, but there are {n_SF1}"

    n_t_SF1 = len(dataset_test_SF1)
    assert (
        n_t_SF1 == 54 * 1
    ), f"dataset_test_SF1 should contains {54*1} datums, but there are {n_t_SF1}"

    print("waveform.py test passed")