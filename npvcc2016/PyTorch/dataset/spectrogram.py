from typing import Callable, List, NamedTuple, Optional, Union
from pathlib import Path
import io

from torch import Tensor, save, load
from torch.utils.data.dataset import Dataset
# currently there is no stub in torchaudio [issue](https://github.com/pytorch/audio/issues/615)
from torchaudio import load as load_wav
from torchaudio.transforms import Spectrogram  # type: ignore

from .waveform import get_dataset_wave_path, preprocess_as_wave
from ...corpus import ItemIdNpVCC2016, Mode, NpVCC2016, Speaker
from ...fs import acquire_zip_fs, save_archive, try_to_acquire_archive_contents


def get_dataset_spec_path(dir_dataset: Path, id: ItemIdNpVCC2016) -> Path:
    return dir_dataset / id.mode / id.speaker / "specs" / f"{id.serial_num}.spec.pt"


def preprocess_as_spec(corpus: NpVCC2016, dir_dataset: Path) -> None:
    """
    Transform npVCC2016 corpus contents into spectrogram Tensor.
    """
    for id in corpus.get_identities():
        waveform, _sr = load_wav(corpus.get_item_path(id))
        # :: [1, Length] -> [Length,]
        waveform: Tensor = waveform[0, :]
        # defaults: hop_length = win_length // 2, window_fn = torch.hann_window, power = 2
        spec: Tensor = Spectrogram(254)(waveform)
        path_spec = get_dataset_spec_path(dir_dataset, id)
        path_spec.parent.mkdir(parents=True, exist_ok=True)
        save(spec, path_spec)


class Datum_NpVCC2016_spec_train(NamedTuple):
    spectrogram: Tensor
    label: str


class Datum_NpVCC2016_spec_test(NamedTuple):
    waveform: Tensor
    spectrogram: Tensor
    label: str


class NpVCC2016_spec(Dataset): # I failed to understand this error
    """
    Audio spectrogram dataset from npVCC2016 non-parallel speech corpus.
    """

    def __init__(
        self,
        train: bool,
        speakers: List[Speaker] = ["SF1", "SM1", "TF2", "TM3"],
        transform: Callable[[Tensor], Tensor] = (lambda i: i),
        download_corpus: bool = False,
        dir_data: str = "./data/",
        corpus_adress: Optional[str] = None,
        dataset_adress: str = "./data/datasets/npVCC2016_spec/archive/dataset.zip",
        zipfs: bool = False,
        compression: bool = True,
        cache: bool = False
    ):
        """
        Args:
            train: train_dataset if True else validation/test_dataset.
            speakers: Selected speaker list.
            transform: Tensor transform on load.
            download_corpus: Whether download the corpus or not when dataset is not found.
            dir_data: Directory in which corpus and dataset are saved.
            corpus_adress: URL/localPath of corpus archive (remote url, like `s3::`, can be used). None use default URL.
            dataset_adress: URL/localPath of dataset archive (remote url, like `s3::`, can be used).
            zipfs: Whether use ZipFileSystem dataset or not (have some performance disadvantage).
            compression: Whether compress dataset or not when new dataset is generated.
        """
        # Design Notes:
        #   Dataset is often saved in the private adress, so there is no `download_dataset` safety flag.
        #   `download` is common option in torchAudio datasets.

        # Store parameters.
        self._train = train
        self._transform = transform
        self._zipfs = zipfs
        self._cache = cache

        # Directory structure:
        # {dir_data}/
        #   corpuses/...
        #   datasets/
        #     npVCC2016_spec/
        #       archive/dataset.zip
        #       contents/{extracted dirs & files}
        self._corpus = NpVCC2016(download_corpus, corpus_adress, f"{dir_data}/corpuses/npVCC2016/")
        self._path_archive_local = Path(dir_data)/"datasets"/"npVCC2016_spec"/"archive"/"dataset.zip"
        self._path_contents_local = Path(dir_data)/"datasets"/"npVCC2016_spec"/"contents"

        # Prepare the dataset.
        mode: Mode = "trains" if train else "evals"
        self._ids: List[ItemIdNpVCC2016] = list(
            filter(lambda id: id.speaker in speakers,
                filter(lambda id: id.mode == mode,
                    self._corpus.get_identities()
        )))
        contents_acquired = try_to_acquire_archive_contents(
            self._path_contents_local,
            self._path_archive_local,
            dataset_adress,
            True
        )
        if not contents_acquired:
            # Generate the dataset contents from corpus
            print("Dataset archive file is not found. Automatically generating new dataset...")
            self._generate_dataset_contents()
            # save dataset archive
            save_archive(
                self._path_contents_local,
                self._path_archive_local,
                dataset_adress,
                compression
            )
            print("Dataset contents was generated and archive was saved.")
        self._fs = acquire_zip_fs(dataset_adress)

        # todo: preprocessing
        # todo: cache
            # if self._cache:
            #     self._data_cache[id.mode][id.speaker][id.serial_num] = spec

    def _generate_dataset_contents(self) -> None:
        """
        Generate dataset with corpus auto-download and preprocessing.
        """
        self._corpus.get_contents()
        preprocess_as_wave(self._corpus, self._path_contents_local)
        preprocess_as_spec(self._corpus, self._path_contents_local)

    def _preprocess_corpus(self):
        # prepare cache dictionary
        self._data_cache = {"trains": {}, "evals": {}}
        # spec directory preparation. directory strucutre: /("evals"|"trains")/(Speaker)/specs/xxxxx.spec
        for mode in ["trains", "evals"]:
            # prepare speaker directories/dictionaries.
            for speaker in ["SF1", "SM1", "TF2", "TM3"]:
                (self._path_corpus / mode / speaker / "specs").mkdir(exist_ok=True)
                self._data_cache[mode][speaker] = {}

    def _load_spec_cache(self, id: ItemIdNpVCC2016) -> Tensor:
        return self._data_cache[id.mode][id.speaker][id.serial_num]

    def _load_datum(self, id: ItemIdNpVCC2016) -> Union[Datum_NpVCC2016_spec_train, Datum_NpVCC2016_spec_test]:
        spec_path = get_dataset_spec_path(self._path_contents_local, id)
        spec: Tensor = self._transform(self._load_spec_cache(id) if self._cache else load(spec_path))
        if self._train:
            return Datum_NpVCC2016_spec_train(spec, f"{id.mode}-{id.speaker}-{id.serial_num}")
        else:
            # todo: cache
            waveform: Tensor = load(get_dataset_wave_path(self._path_contents_local, id))
            return Datum_NpVCC2016_spec_test(waveform, spec, f"{id.mode}-{id.speaker}-{id.serial_num}")

    def _load_datum_from_fs(self, id: ItemIdNpVCC2016) -> Union[Datum_NpVCC2016_spec_train, Datum_NpVCC2016_spec_test]:
        with self._fs.open(get_dataset_spec_path(Path("/"), id), mode="rb") as f:
            spec: Tensor = self._transform(load(io.BytesIO(f.read())))
        if self._train:
            return Datum_NpVCC2016_spec_train(spec, f"{id.mode}-{id.speaker}-{id.serial_num}")
        else:
            with self._fs.open(get_dataset_wave_path(Path("/"), id), mode="rb") as f:
                spec: Tensor = self._transform(load(io.BytesIO(f.read())))
                waveform: Tensor = load(io.BytesIO(f.read()))
            return Datum_NpVCC2016_spec_test(waveform, spec, f"{id.mode}-{id.speaker}-{id.serial_num}")

    def __getitem__(self, n: int) -> Union[Datum_NpVCC2016_spec_train, Datum_NpVCC2016_spec_test]:
        """Load the n-th sample from the dataset.
        Args:
            n : The index of the datum to be loaded
        """
        if self._zipfs:
            return self._load_datum_from_fs(self._ids[n])
        else:
            return self._load_datum(self._ids[n])

    def __len__(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    print("This is spectrogram.py")
    # dataset preparation
    NpVCC2016_spec(train=True, download_corpus=True)  # commented out for safety

    # setup
    dataset_train_SF1 = NpVCC2016_spec(train=True, download_corpus=False, speakers=["SF1"], cache=True)
    print(dataset_train_SF1[0])
    # print(torch.load("./npVCC2016-1.0.0/trains/SF1/specs/100056.spec"))
