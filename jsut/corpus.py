from jsut.fs import forward_from_general
from typing import List, NamedTuple, Optional
from pathlib import Path

from corpuspy.interface import AbstractCorpus
from corpuspy.helper.contents import get_contents
from corpuspy.helper.forward import forward_from_GDrive


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
    """Identity of JSUT corpus's item.
    """
    
    subtype: Subtype
    serial_num: int


class JSUT(AbstractCorpus[ItemIdJSUT]):
    """JSUT corpus.
    
    Archive/contents handler of JSUT corpus.
    """

    gdrive_contents_id: str = "1f7bIQfwWdFOxeaYzs5Cw-HTcA8uwQ8qp"
    
    def __init__(self, adress: Optional[str] = None, download_origin: bool = False) -> None:
        """Initiate JSUT with archive options.

        Args:
            adress: Corpus archive adress (e.g. path, S3) from/to which archive will be read/written through `fsspec`.
            download_origin: Download original corpus when there is no corpus in local and specified adress.
        """

        ver: str = "ver1.1"
        # Equal to 1st layer directory name of original zip.
        self._corpus_name: str = f"jsut_{ver}"
        self._origin_adress = f"http://ss-takashi.sakura.ne.jp/corpus/{self._corpus_name}.zip"

        dir_corpus_local: str = "./data/corpuses/JSUT/"
        default_path_archive = str((Path(dir_corpus_local) / "archive" / f"{self._corpus_name}.zip").resolve())
        self._path_contents_local = Path(dir_corpus_local) / "contents"
        self._adress = adress if adress else default_path_archive

        self._download_origin = download_origin

    def get_contents(self) -> None:
        """Get corpus contents into local.
        """

        get_contents(self._adress, self._path_contents_local, self._download_origin, self.forward_from_origin)

    def forward_from_origin(self) -> None:
        """Forward original corpus archive to the adress.
        """
        # Design Notes:
        #   Do not use http origin because of slow download. Official Google Drive alternative is used.

        forward_from_GDrive(self.gdrive_contents_id, self._adress, 2.5)

    def get_identities(self) -> List[ItemIdJSUT]:
        """Get corpus item identities.

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
        """Get path of the item.

        Args:
            id: Target item identity.
        Returns:
            Path of the specified item.
        """

        subs = {
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
        prefix = subs[id.subtype]
        num = str(id.serial_num).zfill(int(subs[id.subtype]["zfill"]))
        p = f"{root}/{self._corpus_name}/{id.subtype}/wav/{prefix}_{num}.wav"
        return Path(p)