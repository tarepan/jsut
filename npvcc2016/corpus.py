from typing import List, NamedTuple, Optional
from pathlib import Path

import fsspec # type: ignore
from fsspec.utils import get_protocol # type: ignore

from .fs import try_to_acquire_archive_contents


# ## Glossary
# - archive: Single archive file.
# - contents: A directory in which archive's contents exist.


# Mode = Literal["trains", "evals"] # >=Python3.8
Mode = str
# Speaker = Literal["SF1", "SM1", "TF2", "TM3"] # >=Python3.8
Speaker = str


class ItemIdNpVCC2016(NamedTuple):
    mode: Mode
    speaker: Speaker
    serial_num: str


class NpVCC2016:
    def __init__(
        self,
        download: bool = False,
        adress_archive: Optional[str] = None
    ) -> None:
        """
        Wrapper of `npVCC2016` corpus.
        [GitHub](https://github.com/tarepan/npVCC2016Corpus).
        Corpus will be deployed as below.

        {dir_corpus_local}/
            archive/
                f"{corpus_name}.zip"
            contents/
                {extracted dirs & files}

        Args:
        download: Download corpus when there is no archive in local.
        adress_archive: Corpus archive adress (Various url type (e.g. S3, GCP) is accepted through `fsspec` library).
        """
        ver: str = "1.0.0"
        corpus_name: str = f"npVCC2016-{ver}"

        default_url = f"https://github.com/tarepan/npVCC2016Corpus/releases/download/v{ver}/{corpus_name}.zip"
        self._url = adress_archive if adress_archive else default_url
        self._download = download
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(self._url))

        dir_corpus_local: str = "./data/corpuses/npVCC2016/"
        self._path_archive_local = Path(dir_corpus_local) / "archive" / f"{corpus_name}.zip"
        self._path_contents_local = Path(dir_corpus_local) / "contents"

    def get_archive(self) -> None:
        """
        Get the corpus archive file.
        """
        # library selection:
        #   `torchaudio.datasets.utils.download_url` is good for basic purpose, but not compatible with private storages.
        # todo: caching
        path_archive = self._path_archive_local
        if path_archive.exists():
            if path_archive.is_file():
                print("Archive file already exists.")
            else:
                raise RuntimeError(f"{str(path_archive)} should be archive file or empty, but it is directory.")
        else:
            if self._download:
                path_archive.parent.mkdir(parents=True, exist_ok=True)
                self._fs.get_file(self._url, path_archive)
            else:
                raise RuntimeError("Try to get_archive, but `download` is disabled.")

    def get_contents(self) -> None:
        """
        Get the archive and extract the contents if needed.
        """
        # todo: caching
        path_contents = self._path_contents_local
        acquired = try_to_acquire_archive_contents(path_contents, self._url, self._download)
        if not acquired:
            raise RuntimeError(f"Specified corpus archive cannot be acquired. Check the link (`{self._url}`) or `download` option.")

    def get_identities(self) -> List[ItemIdNpVCC2016]:
        """
        Get corpus item identities.

        Returns:
            Full item identity list.
        """
        # data division is described in npVCC2016Corpus GitHub 
        divs = {
            "trains": {
                "SF1": range(100001, 100082), 
                "SM1": range(100001, 100082), 
                "TF2": range(100082, 100163), 
                "TM3": range(100082, 100163)
            },
            "evals": {
                "SF1": range(200001, 200055), 
                "SM1": range(200001, 200055), 
                "TF2": range(200001, 200055), 
                "TM3": range(200001, 200055)
            }
        }
        ids: List[ItemIdNpVCC2016] = []
        for mode in ["trains", "evals"]:
            for speaker in ["SF1", "SM1", "TF2", "TM3"]:
                for num in divs[mode][speaker]:
                    ids.append(ItemIdNpVCC2016(mode, speaker, f"{num}"))
        return ids

    def get_item_path(self, id: ItemIdNpVCC2016) -> Path:
        """
        Get path of the item.

        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        return self._path_contents_local / id.mode / id.speaker / "wavs" / f"{id.serial_num}.wav"
