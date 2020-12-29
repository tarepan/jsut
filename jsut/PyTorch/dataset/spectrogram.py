from typing import Callable, List, NamedTuple, Optional, Union
from pathlib import Path

from torch import Tensor, save, load
from torch.utils.data.dataset import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Spectrogram, Resample # type: ignore
from corpuspy.components.archive import hash_args, save_archive, try_to_acquire_archive_contents

from .waveform import get_dataset_wave_path, preprocess_as_wave
from ...corpus import ItemIdJSUT, Subtype, JSUT


def get_dataset_spec_path(dir_dataset: Path, id: ItemIdJSUT) -> Path:
    return dir_dataset / id.subtype / "specs" / f"{id.serial_num}.spec.pt"



def preprocess_as_spec(path_wav: Path, id: ItemIdJSUT, dir_dataset: Path, new_sr: Optional[int] = None) -> None:
    """Transform JSUT corpus contents into spectrogram Tensor.

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
    """Audio spectrogram dataset from JSUT speech corpus.
    """

    def __init__(
        self,
        train: bool,
        resample_sr: Optional[int],
        subtypes: List[Subtype] = ["basic5000"],
        download_corpus: bool = False,
        corpus_adress: Optional[str] = None,
        dataset_dir_adress: Optional[str] = None,
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            resample_sr: If not None, resample with specified sampling rate.
            subtypes: Corpus item subtypes for the dataset.
            download_corpus: Whether download the corpus or not when dataset is not found.
            corpus_adress: URL/localPath of corpus archive (e.g. `s3::` can be used). None use default URL.
            dataset_dir_adress: URL/localPath of JSUT_spec dataset directory (e.g. `s3::` can be used). None use default local path.
            transform: Tensor transform on load.
        """

        # Design Notes:
        #   Sampling rate:
        #     Sampling rates of dataset A and B should match, so `resample_sr` is not a optional, but required argument.
        #   Download:
        #     Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #     `download` is common option in torchAudio datasets.
        #   Dataset archive name:
        #     Dataset contents differ based on argument, so archive should differ when arguments differ.
        #     It is guaranteed by name by argument hash.

        # Store parameters.
        self._train = train
        self._resample_sr = resample_sr
        self._transform = transform

        self._corpus = JSUT(corpus_adress, download_corpus)
        arg_hash = hash_args(subtypes, resample_sr)
        JSUT_spec_root = Path(".")/"tmp"/"JSUT_spec"
        self._path_contents_local = JSUT_spec_root/"contents"/arg_hash
        dataset_dir_adress = dataset_dir_adress if dataset_dir_adress else str(JSUT_spec_root/"archive")
        dataset_archive_adress = f"{dataset_dir_adress}/{arg_hash}.zip"

        # Prepare data identities.
        self._ids: List[ItemIdJSUT] = list(filter(lambda id: id.subtype in subtypes, self._corpus.get_identities()))

        # Deploy dataset contents.
        contents_acquired = try_to_acquire_archive_contents(dataset_archive_adress, self._path_contents_local)
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            save_archive(self._path_contents_local, dataset_archive_adress)
            print("Dataset contents was generated and archive was saved.")

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and preprocessing.
        """

        self._corpus.get_contents()
        print("Preprocessing...")
        for id in self._ids:
            path_wav = self._corpus.get_item_path(id)
            preprocess_as_spec(path_wav, id, self._path_contents_local, self._resample_sr)
            preprocess_as_wave(path_wav, id, self._path_contents_local, self._resample_sr)
        print("Preprocessed.")

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
    # Subcorpus sample
    from jsut.PyTorch.dataset.spectrogram import JSUT_spec


    JSUT_spec(True, ["basic5000"], download_corpus=True)
    JSUT_spec(True, ["countersuffix26"], download_corpus=True)
    JSUT_spec(True, ["loanword128"], download_corpus=True)
    JSUT_spec(True, ["onomatopee300"], download_corpus=True)
    JSUT_spec(True, ["precedent130"], download_corpus=True)
    JSUT_spec(True, ["travel1000"], download_corpus=True)
    JSUT_spec(True, ["voiceactress100"], download_corpus=True)