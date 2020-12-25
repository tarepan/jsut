import fsspec


def forward_from_general(adress_from: str, forward_to: str) -> None:
    """Forward a file from the adress to specified adress.
    Forward any_adress -> any_adress through fsspec (e.g. local, S3, GCP).

    Args:
        adress_from: Forward origin adress.
        forward_to: Forward distination adress.
    """

    adress_from_with_cache = f"simplecache::{adress_from}" 
    forward_to_with_cache = f"simplecache::{forward_to}"

    with fsspec.open(adress_from_with_cache, "rb") as origin:
        print("Forward: Reading from the adress...")
        archive = origin.read()
        print("Forward: Read.")

        print("Forward: Writing to the adress...")
        with fsspec.open(forward_to_with_cache, "wb") as destination:
            destination.write(archive)
        print("Forward: Written.")


if __name__ == "__main__":
    pass