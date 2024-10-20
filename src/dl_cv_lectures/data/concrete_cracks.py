from pathlib import Path

from . import utils


def download(download_dir: Path):
    utils.download_and_extract_zip(
        url="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip",
        save_path=download_dir.joinpath("concrete.zip"),
        extract_path=download_dir.joinpath("concrete/"),
    )
