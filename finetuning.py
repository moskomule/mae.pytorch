# mainly adopted from https://github.com/facebookresearch/moco
from __future__ import annotations

import builtins

import chika
import homura
import rich
import torch
from homura import reporters
from homura.vision import DATASET_REGISTRY
from torch import nn
from torchvision.transforms import RandAugment, InterpolationMode

from models import ViTModels
from vision_utils import fast_collate, gen_mix_collate


@chika.config
class OptimConfig:
    epochs: int = 100
    warmup_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: list[float] = chika.sequence(0.9, 0.999, size=2)
    label_smoothing: float = 0.1


class DataConfig:
    batch_size: int = None
    no_randaugment: bool = False
    mixup: float = 0.8
    cutmix: float = 1.0

    def __post_init__(self):
        self.randaugment = not self.no_randaugment


@chika.config
class Config:
    optim: OptimConfig
    data: DataConfig

    path: str = chika.required(help="Path to the pretrained weight")
    batch_size: int = 256
    num_workers: int = 32
    finetune_block_ids: list[int] = chika.sequence()
    finetune_all: bool = False

    gpu: int = 0
    seed: int = 0
    amp: bool = False

    def __post_init__(self):
        if self.data.batch_size is not None:
            self.batch_size = self.data.batch_size
        assert not (self.finetune_all and len(self.finetune_block_ids) == 0), \
            'finetune_block_ids and finetune_all are mutually exclusive'
        self.optim.lr *= self.batch_size * homura.get_world_size() / 256


class Trainer(homura.trainers.SupervisedTrainer):
    def __init__(self, *args, **kwargs):
        self.optim_cfg = kwargs.pop('optim_cfg')
        super().__init__(*args, **kwargs)

    def iteration(self,
                  data
                  ) -> None:
        self.model.eval()
        super().iteration(data)

    def set_optimizer(self
                      ) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim._multi_tensor.AdamW(params, lr=self.optim_cfg.lr, betas=self.optim_cfg.betas,
                                                         weight_decay=self.optim_cfg.weight_decay)


def _main(cfg: Config):
    path = (chika.original_path / cfg.path).resolve()
    loaded = torch.load(path, map_location='cpu')
    pretrained_weights: dict[str, torch.Tensor] = loaded['model']
    model = ViTModels(loaded['cfg']['model']['name'])(mean_pooling=True)
    if not cfg.finetune_all:
        learnable_module_names = {'fc.weight', 'fc.bias', 'norm.weight', 'norm.bias'}
        for block_id in cfg.finetune_block_ids:
            if block_id > len(model.blocks):
                raise ValueError(f'Number of blocks in the model is {len(model.blocks)}, but got {block_id}!')
            for name, _ in model.named_parameters():
                if name.startswith(f'blocks.{block_id}'):
                    learnable_module_names.add(name)

        num_learnable_params = 0
        for n, p in model.named_parameters():
            e_n = f'encoder.{n}'
            if pretrained_weights.get(e_n) is not None:
                p.data.copy_(pretrained_weights[e_n].data)
                if n in learnable_module_names:
                    num_learnable_params += 1
                else:
                    p.requires_grad_(False)
        for n, p in model.named_buffers():
            e_n = f'encoder.{n}'
            if pretrained_weights.get(e_n) is not None:
                p.copy_(pretrained_weights[e_n])

        assert len(learnable_module_names) == num_learnable_params, f"mismatch"

    scheduler = homura.lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, cfg.optim.warmup_epochs)
    vs = DATASET_REGISTRY("imagenet")
    vs.collate_fn = fast_collate if cfg.data.mixup + cfg.data.cutmix == 0 else gen_mix_collate(vs.num_classes,
                                                                                               cfg.data.mixup,
                                                                                               cfg.data.cutmix)
    vs.test_collate_fn = fast_collate
    train_da = vs.default_train_da.copy()
    test_da = vs.default_test_da.copy()
    train_da[0].size = model.image_size
    test_da[0].size = model.image_size
    test_da[1].size = model.image_size
    if cfg.data.randaugment:
        train_da.append(RandAugment(interpolation=InterpolationMode.BILINEAR))
    train_loader, test_loader = vs(batch_size=cfg.data.batch_size,
                                   train_da=train_da,
                                   test_da=test_da,
                                   num_workers=cfg.num_workers)

    with Trainer(model, None, nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing), scheduler=scheduler,
                 reporters=[reporters.TensorboardReporter(".")],
                 use_amp=cfg.amp, optim_cfg=cfg.optim) as trainer:
        for epoch in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
            print(f"{epoch=} train acc={trainer.reporter.history('accuracy/train')[-1]:.3f}")
            print(f"{epoch=} test acc@1={trainer.reporter.history('accuracy/test')[-1]:.3f}")


@chika.main(Config, change_job_dir=True)
@homura.distributed_ready_main
def main(cfg):
    if homura.is_master():
        builtins.print = rich.print
    if not homura.is_distributed():
        torch.cuda.set_device(cfg.gpu)
    print(cfg)
    with homura.set_seed(cfg.seed, by_rank=True):
        _main(cfg)


if __name__ == '__main__':
    main()
