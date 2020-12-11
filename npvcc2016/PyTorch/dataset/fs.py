from pathlib import Path
import zipfile
import shutil
from typing import Callable, Optional

import fsspec
from fsspec.utils import get_protocol
from fsspec.implementations.zip import ZipFileSystem


def acquire_dataset_fs(
    generate_ds: Callable[[], str],
    adress: str = "./data/dataset.zip",
    adress_cache: Optional[str] = "./tmp/data/cache",
    compression: bool = True
) -> ZipFileSystem:
    """
    Acquire a filesystem of dataset stored in the adress.
    If a dataset does not exist in specified adress, new dataset is generated and uploaded in the adress.
    Local/Remote adress is accepted through fsspec.
    Automatic zip cache enable efficient remote access.

    generate_ds: Dataset generator which preprocess corpus as a dataset then yeild a dataset root adress.
    """

    # Check existance of dataset archive file in the adress.
    fs = fsspec.filesystem(get_protocol(adress))
    isExist = fs.exists(adress)
    isFile = fs.isfile(adress)

    if isExist and isFile:
        # A dataset file exist, ready to load.
        pass

    elif not isExist:
        print("Dataset file is not found. Automatically generating new dataset...")

        # generate a dataset
        path_dataset_root = generate_ds()
        if compression:
            # compression: deflate
            path_generated = shutil.make_archive(
                f"{adress_cache}/ds/temp", "zip", root_dir=path_dataset_root
            )
        else:
            # compression: no
            path_generated = f"{adress_cache}/ds/temp.zip"
            path_zip_root = Path(path_dataset_root)
            with zipfile.ZipFile(path_generated, mode='w', compression=zipfile.ZIP_STORED) as new_zip:
                for p in path_zip_root.glob("**/*"):
                    if p.is_file():
                        new_zip.write(p, p.relative_to(path_zip_root))

        # write (==upload) the dataset
        with open(path_generated, mode="rb") as stream_zip:
            with fsspec.open(f"simplecache::{adress}", "wb") as f:
                f.write(stream_zip.read())

        print("Dataset zip archive was generated and uploaded.")

    else:
        raise Exception(
            "Directory is in the adress. There should be a file or nothing."
        )

    # Access dataset zip archive and create filesystem
    fs = ZipFileSystem(
        f"simplecache::{adress}",
        # Cache adress is controlable for manual deletion and name collision avoidance.
        target_options={"simplecache": {"cache_storage": adress_cache}},
    )

    return fs


if __name__ == "__main__":

    def thunk():
        return "./static"

    # url_remote = "https://github.com/tarepan/npVCC2016Corpus/releases/download/v1.0.0/npVCC2016-1.0.0.zip"
    url_remote = "s3://tarepan-machine-learning/preprocessingTest/test1.zip"

    # fs = acquire_dataset_fs(thunk, adress="./npVCC2016-f.zip")
    fs = acquire_dataset_fs(thunk, adress=url_remote)
    print(fs.ls("/"))