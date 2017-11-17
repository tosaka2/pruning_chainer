# pruning_chainer
Pruning implemented in chainer

training and model implementation is from https://github.com/chainer/chainer/tree/master/examples/cifar .

## Usage
```python
    # Set pruning
    masks = pruning.create_model_mask(model, args.pruning)
    trainer.extend(pruning.pruned(model, masks))

    # Run the training
    trainer.run()
```