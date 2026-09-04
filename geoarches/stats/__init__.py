"""Resolve external statistics used by geoarches metrics."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from geoarches.download.constants import HUGGINGFACE_REPOSITORY, HUGGINGFACE_REVISION

QUANTILE_FILENAMES = frozenset(
    {
        "era5-quantiles-2016_2022.nc",
        "hres-quantiles-2016_2022.nc",
    }
)
_PACKAGED_STATS_DIRECTORY = Path(__file__).parent


def resolve_quantiles_file(quantiles_filepath: str | Path) -> Path:
    """Return a local quantile file, downloading known statistics when needed.

    Existing local paths take precedence, which lets offline and HPC users pre-stage
    custom statistics. The two standard files are otherwise downloaded once into the
    Hugging Face cache and reused by subsequent runs.
    """
    requested_path = Path(quantiles_filepath).expanduser()
    if requested_path.is_file():
        return requested_path.resolve()

    packaged_path = _PACKAGED_STATS_DIRECTORY / requested_path
    if packaged_path.is_file():
        return packaged_path.resolve()

    if requested_path.parent != Path() or requested_path.name not in QUANTILE_FILENAMES:
        raise FileNotFoundError(
            f"Quantile statistics not found at {requested_path}. Pass an existing local path "
            f"or one of {sorted(QUANTILE_FILENAMES)}."
        )

    return Path(
        hf_hub_download(
            repo_id=HUGGINGFACE_REPOSITORY,
            filename=requested_path.name,
            revision=HUGGINGFACE_REVISION,
        )
    )


__all__ = [
    "QUANTILE_FILENAMES",
    "STATS_REPOSITORY",
    "STATS_REVISION",
    "resolve_quantiles_file",
]
