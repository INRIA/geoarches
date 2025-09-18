"""
Script to run inference made by two or more models and compute average prediction.

Used to store output of ArchesWeather-Mx4 in order to train ArchesWeatherGen on residuals.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from hydra.utils import instantiate
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from geoarches.lightning_modules.base_module import AvgModule, load_module

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="whether to recompute with model")
parser.add_argument(
    "--output-path",
    default="data/outputs/deterministic/archesweather",
    help="where to store outputs",
)
parser.add_argument("--debug", action="store_true", help="whether to debug")
parser.add_argument("--max-lead-time", type=int, default=10, help="max lead time")
parser.add_argument(
    "--uids",
    default="",
    type=str,
    help="Comma separated list of model uids, names of checkpoint folders stored under `modelstore/` dir.",
)

parser.add_argument("--hour", type=int, default=0, help="which hour to run (0, 6, 12, 18)")


args = parser.parse_args()


torch.set_grad_enabled(False)

device = "cuda:0"

model_uids = args.uids.split(",")

if Path(args.output_path).exists() and args.force:
    shutil.rmtree(args.output_path)

Path(args.output_path).mkdir(parents=True, exist_ok=True)


if len(model_uids) > 1:
    module = AvgModule(model_uids).to(device)
    module.eval()
    _, cfg = load_module(model_uids[0])
else:
    module, cfg = load_module(model_uids[0])
    module = module.to(device)
    module.eval()


# create dataset and dataloader
ds = instantiate(
    cfg.dataloader.dataset,
    cfg.stats,
    domain=f"all_from_1979_{args.hour}h",
)

p = cfg.dataloader.dataset.path
era5_p  = p.split("/")[-2]
print("ERA5 path:", era5_p)
def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}


dl = torch.utils.data.DataLoader(
    ds, batch_size=1, num_workers=3, shuffle=False, collate_fn=collate_fn
)

current_year = 1979
max_year = 1982
xr_list = []

for i, batch in tqdm(enumerate(dl)):
    fname = Path(args.output_path).joinpath(f"{era5_p}_pred_{current_year}_0h.nc")
    if fname.exists():
        continue

    next_year = pd.to_datetime(batch["timestamp"][0] + 6 * 3600, utc=True, unit="s").year
    print(pd.to_datetime(batch["timestamp"][0], utc=True, unit="s").tz_localize(None), next_year, current_year)
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for (k, v) in batch.items()}
    batch = {k: (v.float() if 'state' in k else v) for (k, v) in batch.items()}
    out = module.forward(batch)
    denorm_out = ds.denormalize(out)

    xr_list.append(ds.convert_to_xarray(denorm_out, batch["timestamp"]))

    if next_year > current_year:
        print("saving file", current_year)
        # save outputs
        xr_dataset = xr.concat(xr_list, dim="time")

        fname = f"{era5_p}_pred_{current_year}_{hour:02d}h.nc"
        xr_dataset.sel(time=(xr_dataset.time.dt.hour == hour)).to_netcdf(
            Path(args.output_path) / fname,
            encoding=dict(time=dict(units="hours since 2000-01-01")) if not i else None,
        )
        xr_list.clear()
        current_year = next_year

        if current_year > max_year:
            break
