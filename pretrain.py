import chika
import homura
import torch
from homura import lr_scheduler, reporters
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY
from torch.nn import functional as F

from models import MaskedAutoEncoder, ViTModels

try:
    import rich
    import builtins

    builtins.print = rich.print
except ImportError:
    pass


class ViTTrainer(SupervisedTrainer):
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs.pop('cfg')
        super().__init__(*args, **kwargs)

    def set_optimizer(self
                      ) -> None:
        optim_cfg = self.cfg.optim
        kwargs = dict(lr=optim_cfg.lr, betas=optim_cfg.betas, weight_decay=optim_cfg.weight_decay)
        params_dict = self.accessible_model.param_groups
        optim_groups = [
            {"params": params_dict['decay'], "weight_decay": optim_cfg.weight_decay},
            {"params": params_dict['no_decay'], "weight_decay": 0}
        ]
        optim = torch.optim._multi_tensor.AdamW
        self.optimizer = optim(optim_groups, **kwargs)
        self.logger.debug(self.optimizer)

    def iteration(self,
                  data
                  ) -> None:
        input, target = data
        with torch.cuda.amp.autocast(self._use_amp):
            pred_patch, gt_patch = self.model(input)
            loss = F.mse_loss(pred_patch, gt_patch)

        if self.is_train:
            self.optimizer.zero_grad()
            if self._use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        self.reporter.add('loss', loss.detach_())

    def state_dict(self
                   ):
        d = super().state_dict()
        d['cfg'] = self.cfg.to_dict()
        return d


@chika.config
class DataConfig:
    batch_size: int = 4_096


@chika.config
class ModelConfig:
    name: str = chika.choices(*ViTModels.choices())
    dec_emb_dim: int = 128
    dec_depth: int = 4
    dec_num_heads: int = 8
    mask_ratio: float = 0.75


@chika.config
class OptimConfig:
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    epochs: int = 200
    min_lr: float = 1e-6
    warmup_epochs: int = 20
    betas: list[float] = chika.sequence(0.9, 0.95, size=2)


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    num_workers: int = 8
    debug: bool = False
    amp: bool = False
    gpu: int = None

    def __post_init__(self):
        assert self.optim.lr > self.optim.min_lr
        adjuster = self.data.batch_size * homura.get_world_size() / 256
        self.optim.lr *= adjuster
        self.optim.min_lr *= adjuster


@chika.main(cfg_cls=Config, change_job_dir=True)
@homura.distributed_ready_main
def main(cfg: Config):
    if cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
    if homura.is_master():
        import rich
        rich.print(cfg)
    vs = DATASET_REGISTRY("imagenet")
    encoder = ViTModels(cfg.model.name)(mean_pooling=True)
    model = MaskedAutoEncoder(encoder, cfg.model.dec_emb_dim, cfg.model.dec_depth, cfg.model.dec_num_heads,
                              cfg.model.mask_ratio)
    train_loader, _ = vs(batch_size=cfg.data.batch_size,
                         train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                         test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                         num_workers=cfg.num_workers)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       warmup_epochs=cfg.optim.warmup_epochs,
                                                       min_lr=cfg.optim.min_lr)

    with ViTTrainer(model, None, None,
                    reporters=[reporters.TensorboardReporter(".")],
                    scheduler=scheduler,
                    use_amp=cfg.amp,
                    use_cuda_nonblocking=True,
                    cfg=cfg,
                    debug=cfg.debug,
                    dist_kwargs=dict(find_unused_parameters=True)
                    ) as trainer:
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.scheduler.step()
            trainer.save(f"checkpoints", f"{ep}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
