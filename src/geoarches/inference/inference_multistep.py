# script to evaluate a prediction made by a model
# currently not used, prefer to use test_step function by calling
# python main_hydra.py ++mode=test
import argparse
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from hydra.utils import instantiate
from tqdm import tqdm

from geoarches.dataloaders import era5
from geoarches.lightning_modules import load_module
from geoarches.evaluation import headline_wrmse

parser = argparse.ArgumentParser()
parser.add_argument("--uid", default="", type=str, help="model uid")
parser.add_argument("--multistep", default=10, type=int, help="multistep")
parser.add_argument("--step", default="last", type=str, help="model uid")
parser.add_argument("--force", action="store_true", help="whether to recompute with model")
parser.add_argument("--debug", action="store_true", help="whether to debug")


args = parser.parse_args()


torch.set_grad_enabled(False)

file_path = Path(f"evalstore/{args.uid}/test2020-step={args.step}-multistep.zarr")
Path(file_path).parent.mkdir(parents=True, exist_ok=True)

if file_path.exists():
    if not args.force:
        print("output already exists. Exiting..")
        exit()
    else:
        shutil.rmtree(file_path)


module, cfg = load_module(f"modelstore/{args.uid}")

ds_test = instantiate(cfg.dataloader.dataset, domain="test", z0012=True, multistep=args.multistep)
test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=1, num_workers=4, shuffle=False)

err_log = []
buffer_preds_future = []
write_frequency = 10

for i, batch in tqdm(enumerate(test_dataloader)):
    preds_future = module.forward_multistep(batch, args.multistep, return_dict="list")
    buffer_preds_future.append(preds_future)

    err_log.append(
        headline_wrmse(
            preds_future, batch["future_states"], denormalize_function=ds_test.denormalize
        )
    )

    if args.debug:
        breakpoint()

    if not i % write_frequency or i == len(ds_test) - 1:
        pred_list = [
            torch.cat([x[i] for x in buffer_preds_future], dim=0) for i in range(args.multistep)
        ]  # cat on time
        xr_timedelta_list = [era5.convert_to_xarray(p) for p in pred_list]
        prediction_timedeltas = [timedelta(i) for i in range(1, args.multistep + 1)]
        merged_xr_dataset = xr.concat(
            xr_timedelta_list, pd.Index(prediction_timedeltas, name="prediction_timedelta")
        )

        merged_xr_dataset.to_zarr(
            file_path,
            append_dim="time" if i else None,
            encoding=dict(time=dict(units="hours since 2000-01-01")) if not i else None,
        )
        buffer_preds_future = []

        avg_err = {k: torch.cat([e[k] for e in err_log], dim=0).mean(0) for k in err_log[0].keys()}
        print("errors", avg_err)

    if args.debug:
        breakpoint()

avg_err = {
    k: torch.cat([e[k] for e in err_log], dim=0).mean(0) for k in err_log[0].keys()
}  # avg on first dimension which is batch.
torch.save(avg_err, file_path.parent.joinpath(f"allmetrics-multistep-step={args.step}metrics.pt"))
