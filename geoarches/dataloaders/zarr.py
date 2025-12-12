import shutil

import fasteners
import xarray as xr


class _NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ZarrIterativeWriter:
    def __init__(self, path, force=True, use_synchronizer=True):
        """
        Args:
            path: The path to the Zarr dataset.
            force: Whether to delete the dataset if it already exists.
            use_synchronizer: Whether to use a synchronizer to ensure thread-safety.
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if use_synchronizer:
            self.synchronizer = fasteners.InterProcessLock(path.parent / ".lock")
        else:
            self.synchronizer = _NoOpContext()
        print("Creating ZarrIterativeWriter")
        with self.synchronizer:
            if path.exists() and force:
                print("path already exists")
                shutil.rmtree(path)

    def write(self, xr_dataset, step: int | None = None, append_dim="time"):
        with self.synchronizer:
            is_initialized = self.path.exists()
            args = (
                dict(append_dim=append_dim)
                if is_initialized
                else dict(encoding=dict({append_dim: dict(units="hours since 2000-01-01")}))
            )

            xr_dataset.to_zarr(self.path, **args)

    def to_netcdf(self, dump_id=0):
        """
        useful for not hitting inodes limit and rsyncing to remote
        this has to be called once in the main process, otherwise every process will convert to netcdf
        """
        with self.synchronizer:
            nc_path = self.path.parent / self.path.name.replace(".zarr", f"_{dump_id}.nc")
            if not nc_path.exists():
                xr.open_zarr(self.path).to_netcdf(nc_path)
                shutil.rmtree(self.path)
