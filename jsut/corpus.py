from typing import List, NamedTuple, Optional
from pathlib import Path

import fsspec # type: ignore
from fsspec.utils import get_protocol # type: ignore

from .fs import try_to_acquire_archive_contents


# ## Glossary
# - archive: Single archive file.
# - contents: A directory in which archive's contents exist.


# Mode = Literal[Longform, Shortform, "simplification", "summarization"] # >=Python3.8
Subtype = str
subtypes = [
    "basic5000",
    "countersuffix26",
    "loanword128",
    "onomatopee300",
    "precedent130",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100",
]


class ItemIdJSUT(NamedTuple):
    subtype: Subtype
    serial_num: int


class JSUT:
    def __init__(
        self,
        download: bool = False,
        adress_archive: Optional[str] = None
    ) -> None:
        """
        Wrapper of `jsut` corpus.
        [Website](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).
        Corpus will be deployed as below.

        Args:
            download: Download corpus when there is no archive in local.
            adress_archive: Corpus archive adress (Various url type (e.g. S3, GCP) is accepted through `fsspec` library).
        """
        ver: str = "ver1.1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"jsut_{ver}"

        default_url = f"http://ss-takashi.sakura.ne.jp/corpus/{self._corpus_name}.zip"
        self._url = adress_archive if adress_archive else default_url
        self._download = download
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(self._url if self._url else "./"))

        dir_corpus_local: str = "./data/corpuses/jsut/"
        self._path_archive_local = Path(dir_corpus_local) / "archive" / f"{self._corpus_name}.zip"
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

    def get_identities(self) -> List[ItemIdJSUT]:
        """
        Get corpus item identities.

        Returns:
            Full item identity list.
        """
        subtype_info = {
            "basic5000": range(1, 5001),
            "countersuffix26": range(1, 27),
            "loanword128": range(1, 129),
            "onomatopee300": range(1, 301),
            "precedent130": range(1, 131),
            "repeat500": range(1, 101),
            "travel1000": range(1, 1001),
            "utparaphrase512": range(1, 315),
            "voiceactress100": range(1, 101),
        }
        ids: List[ItemIdJSUT] = []
        for subtype in subtypes:
            for num in subtype_info[subtype]:
                ids.append(ItemIdJSUT(subtype, num))
        return ids

    def get_item_path(self, id: ItemIdJSUT) -> Path:
        """
        Get path of the item.

        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """
        subtype_dict = {
            "basic5000": {"prefix": "BASIC5000", "zfill": 4},
            "countersuffix26": {"prefix": "COUNTERSUFFIX26", "zfill": 2},
            "loanword128": {"prefix": "LOANWORD128", "zfill": 3},
            "onomatopee300": {"prefix": "ONOMATOPEE300", "zfill": 3},
            "precedent130": {"prefix": "PRECEDENT130", "zfill": 3},
            "repeat500": {"prefix": "REPEAT500_setN", "zfill": 3},
            "travel1000": {"prefix": "TRAVEL1000", "zfill": 4},
            "utparaphrase512": {"prefix": "UT-PARAPHRASE-sentXXX-phraseY", "zfill": 3},
            "voiceactress100": {"prefix": "VOICEACTRESS100", "zfill": 3},
        }
        root = str(self._path_contents_local)
        prefix = subtype_dict[id.subtype]
        num = str(id.serial_num).zfill(int(subtype_dict[id.subtype]["zfill"]))
        p = f"{root}/{self._corpus_name}/{id.subtype}/wav/{prefix}_{num}.wav"
        return Path(p)