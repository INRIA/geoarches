import hydra
import omegaconf

with hydra.initialize(version_base=None, config_path="../../geoarches/configs", job_name="test"):
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "module=archesweathergen",
            "module.module.load_deterministic_model=Null",
            "module.inference.num_steps=1",
            "module.embedder.emb_dim=48",
            "module.embedder.out_emb_dim=96",
            "module.backbone.emb_dim=48",
        ],
    )
    omegaconf.OmegaConf.resolve(cfg)
