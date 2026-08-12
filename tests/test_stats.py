import pytest

from geoarches import stats


def test_resolve_quantiles_file_prefers_existing_path(tmp_path, monkeypatch):
    local_file = tmp_path / "custom-quantiles.nc"
    local_file.touch()

    def fail_download(**kwargs):
        pytest.fail(f"Unexpected download: {kwargs}")

    monkeypatch.setattr(stats, "hf_hub_download", fail_download)

    assert stats.resolve_quantiles_file(local_file) == local_file.resolve()


def test_resolve_quantiles_file_downloads_known_file(tmp_path, monkeypatch):
    cached_file = tmp_path / "era5-quantiles-2016_2022.nc"
    cached_file.touch()
    download_arguments = {}

    def fake_download(**kwargs):
        download_arguments.update(kwargs)
        return str(cached_file)

    monkeypatch.setattr(stats, "hf_hub_download", fake_download)
    monkeypatch.setattr(stats, "_PACKAGED_STATS_DIRECTORY", tmp_path / "package")

    result = stats.resolve_quantiles_file("era5-quantiles-2016_2022.nc")

    assert result == cached_file
    assert download_arguments == {
        "repo_id": stats.STATS_REPOSITORY,
        "filename": "era5-quantiles-2016_2022.nc",
        "revision": stats.STATS_REVISION,
    }


def test_resolve_quantiles_file_rejects_missing_custom_path(tmp_path):
    missing_file = tmp_path / "custom-quantiles.nc"

    with pytest.raises(FileNotFoundError, match="Pass an existing local path"):
        stats.resolve_quantiles_file(missing_file)
