# mainly adopted from https://github.com/facebookresearch/moco
from __future__ import annotations

import builtins

import chika
import homura
import rich
import torch
from homura.vision import DATASET_REGISTRY
from torch.nn import functional as F

from models import vit_b16


@chika.config
class OptimConfig:
    epochs: int = 90
    warmup_epochs: int = 10
    lr: float = 0.1
    weight_decay: float = 0


@chika.config
class Config:
    optim: OptimConfig
    path: str = chika.required(help="Path to the pretrained weight")
    batch_size: int = 256
    num_workers: int = 32
    finetune_block_ids: list[int] = chika.sequence()

    gpu: int = 0
    seed: int = 0
    amp: bool = False

    def __post_init__(self):
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
        self.optimizer = torch.optim.SGD(params, lr=self.optim_cfg.lr, momentum=0.9,
                                         weight_decay=self.optim_cfg.weight_decay)


def _main(cfg: Config):
    pretrained_weights: dict[str, torch.Tensor] = torch.load(cfg.path, map_location='cpu')['model']
    model = vit_b16(mean_pooling=True)
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

    assert len(learnable_module_names) == num_learnable_params, f"{learnable_module_names=}, {num_learnable_params=}"

    scheduler = homura.lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, cfg.optim.warmup_epochs)
    train_loader, test_loader = DATASET_REGISTRY('imagenet')(batch_size=cfg.batch_size,
                                                             num_workers=cfg.num_workers)

    with Trainer(model, None, F.cross_entropy, scheduler=scheduler,
                 use_amp=cfg.amp, optim_cfg=cfg.optim) as trainer:
        for epoch in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
            print(f"{epoch=} train acc={trainer.reporter.history('accuracy/train')[-1]:.3f}")
            print(f"{epoch=} test acc@1={trainer.reporter.history('accuracy/test')[-1]:.3f}")

        print(trainer.compute_conf_matrix(test_loader))


@chika.main(Config)
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