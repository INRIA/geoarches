# Evaluate

Refer to the general [user guide](../user_guide/evaluate.md) for evaluation.

!!! tip

    To run ArchesWeatherGen, it might be useful to first cache the inference outputs of ArchesWeatherMx4 and then pass in `pred_path` with `++dataloader.dataset.pred_path=...`. Otherwise, deterministic models will be loaded and evaluated during (Models to load are specified in the hydra config under `module.load_deterministic_model`). Ensemble predictions for ArchesWeatherMx4can be made with `geoarches/inference/encode_dataset.py`.
