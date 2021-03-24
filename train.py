import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    model = instantiate(cfg.pipeline, cfg=cfg)
    trainer = pl.Trainer(**cfg.pl_trainer)
    datamodule = instantiate(cfg.dataset)
    trainer.fit(model, datamodule)
    if cfg.run_test:
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
