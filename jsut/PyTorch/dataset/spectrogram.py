from typing import Callable, List, NamedTuple, Optional, Union
from pathlib import Path

from torch import Tensor, save, load
from torch.utils.data.dataset import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Spectrogram, Resample # type: ignore

from .waveform import get_dataset_wave_path, preprocess_as_wave
from ...corpus import ItemIdJSUT, Subtype, JSUT
from ...fs import hash_args, save_archive, try_to_acquire_archive_contents


def get_dataset_spec_path(dir_dataset: Path, id: ItemIdJSUT) -> Path:
    return dir_dataset / id.subtype / "specs" / f"{id.serial_num}.spec.pt"


def preprocess_as_spec(corpus: JSUT, dir_dataset: Path, new_sr: Optional[int] = None) -> None:
    """
    Transform npVCC2016 corpus contents into spectrogram Tensor.

    Args:
        new_sr: If specified, resample with specified sampling rate.
    """
    for id in corpus.get_identities():
        waveform, _sr_orig = load_wav(corpus.get_item_path(id))
        if new_sr is not None:
            waveform = Resample(_sr_orig, new_sr)(waveform)
        # :: [1, Length] -> [Length,]
        waveform: Tensor = waveform[0, :]
        # defaults: hop_length = win_length // 2, window_fn = torch.hann_window, power = 2
        spec: Tensor = Spectrogram(254)(waveform)
        path_spec = get_dataset_spec_path(dir_dataset, id)
        path_spec.parent.mkdir(parents=True, exist_ok=True)
        save(spec, path_spec)


class Datum_JSUT_spec_train(NamedTuple):
    spectrogram: Tensor
    label: str


class Datum_JSUT_spec_test(NamedTuple):
    waveform: Tensor
    spectrogram: Tensor
    label: str


class JSUT_spec(Dataset): # I failed to understand this error
    """
    Audio spectrogram dataset from npVCC2016 non-parallel speech corpus.
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
        self._train = train
        self._resample_sr = resample_sr
        self._transform = transform

        self._corpus = JSUT(download_corpus, corpus_adress)
        dirname = hash_args(train, subtypes, download_corpus, corpus_adress, dataset_adress)
        JSUT_spec_root = Path(".")/"tmp"/"JSUT_spec"
        self._path_contents_local = JSUT_spec_root/"contents"/dirname
        dataset_adress = dataset_adress if dataset_adress else str(JSUT_spec_root/"archive"/f"{dirname}.zip")

        # Prepare data identities.
        self._ids: List[ItemIdJSUT] = list(filter(lambda id: id.subtype in subtypes, self._corpus.get_identities()))

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
        preprocess_as_spec(self._corpus, self._path_contents_local, self._resample_sr)

    def _load_datum(self, id: ItemIdJSUT) -> Union[Datum_JSUT_spec_train, Datum_JSUT_spec_test]:
        spec_path = get_dataset_spec_path(self._path_contents_local, id)
        spec: Tensor = load(spec_path)
        spec = self._transform(spec)
        label = f"{id.subtype}-{id.serial_num}"
        if self._train:
            return Datum_JSUT_spec_train(spec, label)
        else:
            waveform: Tensor = load(get_dataset_wave_path(self._path_contents_local, id))
            return Datum_JSUT_spec_test(waveform, spec, label)

    def __getitem__(self, n: int) -> Union[Datum_JSUT_spec_train, Datum_JSUT_spec_test]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    print("This is spectrogram.py")
    # dataset preparation
    JSUT_spec(train=True, download_corpus=True)  # commented out for safety

    # setup
    dataset_train_SF1 = JSUT_spec(train=True, download_corpus=False)
    print(dataset_train_SF1[0])
    # print(torch.load("./npVCC2016-1.0.0/trains/SF1/specs/100056.spec"))
