import hydra
import omegaconf

with hydra.initialize(version_base="1.2", config_path="../../geoarches/configs"):
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "module.embedder.emb_dim=48",
            "module.embedder.out_emb_dim=96",
            "module.backbone.emb_dim=48",
        ],
    )

    omegaconf.OmegaConf.resolve(cfg)
