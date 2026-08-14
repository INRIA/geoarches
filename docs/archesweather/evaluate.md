# Evaluate

Refer to the general [user guide](../user_guide/evaluate.md) for evaluation.

!!! tip

    To run ArchesWeatherGen, it might be useful to first cache the inference outputs of
    ArchesWeatherMx4 and then pass `++dataloader.dataset.pred_path=...`. Otherwise, the
    deterministic models listed under `module.load_deterministic_model` in the Hydra config
    are loaded during evaluation. Generate ArchesWeatherMx4 ensemble predictions with
    `python -m geoarches.inference.encode_dataset`.
