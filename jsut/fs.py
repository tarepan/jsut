import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory

import fsspec
from fsspec.utils import get_protocol
from torchaudio.datasets.utils import extract_archive


def try_to_acquire_archive_contents(
    path_contents_local: Path,
    adress_archive: str,
    download: bool = False
) -> bool:
    """
    Try to acquire the contents of the archive.
    Priority:
      1. (already extracted) local contents
      2. adress-specified (local|remote) archive or its cache through fsspec

    Returns:
        True if success_acquisition else False
    """
    # validation
    if path_contents_local.is_file():
        raise RuntimeError(f"contents ({str(path_contents_local)}) should be directory or empty, but it is file.")

    if path_contents_local.exists():
        # contents directory already exists.
        return True
    else:
        fs: fsspec.AbstractFileSystem = fsspec.filesystem(get_protocol(adress_archive))
        archiveExists = fs.exists(adress_archive)
        archiveIsFile = fs.isfile(adress_archive)

        if archiveExists and (not archiveIsFile):
            raise RuntimeError(f"Archive ({adress_archive}) should be file or empty, but it is directory.")

        adress_archive = f"simplecache::{adress_archive}"
        if archiveExists and download:
            # A dataset file exist, so pull and extract.
            with fsspec.open(adress_archive, "rb") as archive:
                with NamedTemporaryFile("wb") as tmp:
                    tmp.write(archive.read())
                    tmp.seek(0)
                    extract_archive(tmp.name, str(path_contents_local))
            return True
        else:
            # no corresponding archive. Failed to acquire.
            return False


def save_archive(path_contents: Path, adress_archive: str) -> None:
    """
    Save contents as ZIP archive.

    Args:
        path_contents: Contents root directory path
        adress_archive: Saved adress
    """
    with TemporaryDirectory() as tmpdir:
        # zip with deflate compression
        shutil.make_archive(f"{tmpdir}/tmp", "zip", root_dir=path_contents)
        # write (==upload) the archive
        with fsspec.open(f"simplecache::{adress_archive}", "wb") as target:
            with open(f"{tmpdir}/tmp.zip", "rb") as archive:
                target.write(archive.read())

def hash_args(*args) -> str:
    contents = ""
    for c in args:
        contents = f"{contents}_{str(c)}"
    contents = hashlib.md5(contents.encode('utf-8')).hexdigest()
    return contents

if __name__ == "__main__":
    print(hash_args(1,2,3,4,5))
    print(hash_args(1,2,3,4,6))