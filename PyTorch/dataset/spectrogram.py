from typing import Callable, List, NamedTuple, Union
from pathlib import Path

import torch
from torch import Tensor, save

# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
import torchaudio  # type: ignore
from torchaudio.transforms import Spectrogram  # type: ignore

from . import waveform

# Datum_identity, NpVCC2016, Speaker


class Datum_NpVCC2016_spec_train(NamedTuple):
    spectrogram: Tensor
    label: str


class Datum_NpVCC2016_spec_test(NamedTuple):
    waveform: Tensor
    spectrogram: Tensor
    label: str


class NpVCC2016_spec(waveform.NpVCC2016):
    """
    Audio linear spectrogram dataset from NpVCC2016 non-parallel speech corpus
    Args:
        root: Root directory where the dataset's top level directory is found.
        train: train ? train_dataset : test_dataset
        download: Whether to download the dataset if it is not found at root path (common in torchAudio datasets).
        speakers: selection of npVCC2016 speakers for usage
        transform: transform on load
    """

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool = False,
        speakers: List[waveform.Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
    ):
        # preparation + preprocessing with super init
        super().__init__(root, train, download, speakers, transform)

    def _preprocess_corpus(self):
        """
        Preprocess corpus waveform into spectrogram
        """
        # spec directory preparation. directory strucutre: /("evals"|"trains")/(Speaker)/specs/xxxxx.spec
        for te in ["trains", "evals"]:
            # prepare directories.
            for speaker in ["SF1", "SM1", "TF2", "TM3"]:
                (self._path_corpus / te / speaker / "specs").mkdir(exist_ok=True)

        # wave2spec
        ids = self._corpus_item_identities
        for id in ids:
            wave: Tensor
            # no stub problem (see import parts) + torchaudio internal override (It is my guess. It looks like no-interface problem?)
            # pylint: disable=no-member
            wave, _sr = torchaudio.load(self._calc_path_wav(self._path_corpus, id))  # type: ignore
            p = self._calc_path_spec(self._path_corpus, id)
            # defaults: hop_length = win_length // 2, window_fn = torch.hann_window, power = 2
            spec: Tensor = Spectrogram(254)(wave)  # type cannot be inferred
            save(spec, p)

    def _calc_path_spec(self, path_corpus: Path, id: waveform.Datum_identity) -> Path:
        return path_corpus / id.mode / id.speaker / "specs" / f"{id.serial_num}.spec"

    def _load_datum(
        self, path_corpus: Path, id: waveform.Datum_identity
    ) -> Union[Datum_NpVCC2016_spec_train, Datum_NpVCC2016_spec_test]:
        spec_path = self._calc_path_spec(path_corpus, id)
        # no stub problem (see import parts) + torchaudio internal override (It is my guess. It looks like no-interface problem?)
        # pylint: disable=no-member
        spec: Tensor = torch.load(spec_path)  # type: ignore
        if self._train:
            return Datum_NpVCC2016_spec_train(
                spec, f"{id.mode}-{id.speaker}-{id.serial_num}"
            )
        else:
            # no stub problem (see import parts) + torchaudio internal override (It is my guess. It looks like no-interface problem?)
            waveform: Tensor
            # pylint: disable=no-member
            waveform, _sr = torchaudio.load(self._calc_path_wav(path_corpus, id))  # type: ignore
            return Datum_NpVCC2016_spec_test(
                waveform, spec, f"{id.mode}-{id.speaker}-{id.serial_num}"
            )

    def __getitem__(
        self, n: int
    ) -> Union[Datum_NpVCC2016_spec_train, Datum_NpVCC2016_spec_test]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        return self._load_datum(self._path_corpus, self._datum_identities[n])


if __name__ == "__main__":
    print("This is spectrogram.py")
    # dataset preparation
    NpVCC2016_spec(".", train=True, download=True)  # commented out for safety

    # setup
    dataset_train_SF1 = NpVCC2016_spec(
        ".", train=True, download=False, speakers=["SF1"]
    )
    print(dataset_train_SF1[0])
    print(torch.load("./npVCC2016-1.0.0/trains/SF1/specs/100056.spec"))
