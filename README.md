# mae.pytorch

A PyTorch implementation of Masked Autoencoders (WIP)

## Requirements

```
Python>=3.9
PyTorch>=1.10
torchvision
homura-core
chika
rich
```

## Training

### Pre-training

```shell
python [-m torch.distributed.run --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS}] pretrain.py [--amp]
```

### Finetuning

#### Linear finetuning

```shell
python linear_finetuning.py --path ${PATH_TO_PRETRAINED_WEIGHT}
```

#### End-to-end finetuning

```shell
python finetuning.py --path ${PATH_TO_PRETRAINED_WEIGHT} [--finetune_block_ids ...] [--finetune_all]
```

## Results

|                                  | Vit-B           |
|----------------------------------|-----------------|
| Accuracy (Linear probing)        | 54.5 (BS=8,192) |
| Accuracy (End-to-end finetuning) | -               |
