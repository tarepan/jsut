from typing import Optional

import fsspec # type: ignore
from fsspec.utils import get_protocol # type: ignore
from torchaudio.datasets.utils import extract_archive # type: ignore

# Speaker = Literal["SF1", "SM1", "TF2", "TM3"] # >=Python3.8
Speaker = str


class NpVCC2016:
    def __init__(self, url: Optional[str]) -> None:
        self._a = "a"
        ver: str = "1.0.0"
        corpus_name: str = f"npVCC2016-{ver}"

        self._url = url if url else f"https://github.com/tarepan/npVCC2016Corpus/releases/download/v{ver}/{corpus_name}.zip"
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(self._url))
        self._file_name_base = corpus_name
        self._file_name = f"{self._file_name_base}.zip"

    def get_file(self, dir_save: str = "./") -> None:
        """
        Get the npVCC2016 corpus file.
        Various url type (e.g. S3, GCP) is accepted through `fsspec` library.

        Args:
            dir_save: file-saved directory
        """
        # library selection:
        #   `torchaudio.datasets.utils.download_url` is good for basic purpose, but not compatible with private storages.
        # todo: caching
        self._fs.get_file(self._url, dir_save)

    def get_contents(self, dir_extracted: Optional[str] = None) -> None:
        dir_saved = "./tmp/data/corpus/"
        self.get_file(dir_saved)

        dir_extracted = dir_extracted if dir_extracted else f"./data/corpus/{self._file_name_base}/"
        extract_archive(f"{dir_saved}/{self._file_name}", dir_extracted)
