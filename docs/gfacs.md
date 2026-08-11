## GFACS with NLS for CVRPTW

### Dataset Generation
```raw
$ python utils.py
```
This will take up about 2GB of space.

Generate VRPTW datasets by setting all customer demands to zero.
```raw
$ python utils.py --vrptw
```


### Training

The checkpoints will be saved in [`../pretrained/cvrptw`](../pretrained/cvrptw).

Train GFACS model for CVRPTW with `$N` nodes
```raw
$ python train.py $N
```

Train GFACS model for VRPTW with `$N` nodes
```raw
$ python train.py $N --vrptw
```


Train a DeepACO-style REINFORCE model while keeping the GFACS ACO sampler/inference for CVRPTW:
```raw
$ python train_deepaco.py $N --disable_wandb
```

Use local-search-improved costs as the REINFORCE reward, similar to DeepACO-NLS:
```raw
$ python train_deepaco.py $N --use_ls_reward --disable_wandb
```

### Testing

Test GFACS for CVRPTW with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```

Test GFACS for VRPTW with `$N` nodes
```raw
$ python test.py $N --vrptw -p "path_to_checkpoint"
```
