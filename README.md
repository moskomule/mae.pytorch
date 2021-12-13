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

### Fine tuning

```shell
python finetuning.py --path ${PATH_TO_PRETRAINED_WEIGHT} [--finetune_block_ids ...]
```

## Results

|                                | Vit-B [^0] |
|--------------------------------|------------|
| Accuracy (Linear probing [^1]) |            |

[^0]: Trained on ImageNet 

[^1]: 