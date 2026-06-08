# Evaluate

Refer to the general [user guide](../user_guide/evaluate.md) for evaluation.

!!! info For faster inference of ArchesWeatherGen, it's might be useful to cache the inference outputs of ArchesWeatherMx4 first and then pass in `pred_path` with `++dataloader.dataset.pred_path=...`. Otherwise deterministic models are loaded during inference, which is specified in the hydra confug under `module.load_deterministic_model` Ensemble predictions can be made with `geoarches/inference/encode_dataset.py`.
