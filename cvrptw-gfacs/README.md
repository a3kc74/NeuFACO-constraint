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


### Testing

Test GFACS for CVRPTW with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```

Test GFACS for VRPTW with `$N` nodes
```raw
$ python test.py $N --vrptw -p "path_to_checkpoint"
```
