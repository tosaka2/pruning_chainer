import chainer
import chainer.links as L
from chainer import training

import numpy as np

def create_layer_mask(layer, pruning_rate):
    if layer.W == None:
        return

    abs_W = np.abs(layer.W.data)
    data = np.sort(np.ndarray.flatten(abs_W))
    num_prune = int(len(data) * pruning_rate)
    idx_prune = min(num_prune, len(data)-1)
    threshould = data[idx_prune]

    mask = abs_W
    mask[mask < threshould] = 0
    mask[mask >= threshould] = 1
    return mask

'''Returns a trainer extension to fix pruned weight of the model.
'''
def create_model_mask(model, pruning_rate):
    masks = {}
    for name, link in model.namedlinks():
        # specify pruned layer
        if type(link) not in (L.Convolution2D, L.Linear):
            continue
        mask = create_layer_mask(link, pruning_rate)
        masks[name] = mask
    return masks

def prune_weight(model, masks):
    for name, link in model.namedlinks():
        if name not in masks.keys():
            continue
        mask = masks[name]
        link.W.data = link.W.data * mask

'''Returns a trainer extension to fix pruned weight of the model.
'''
def pruned(model, masks):
    @training.make_extension(trigger=(1, 'epoch'))
    def _pruned():
        prune_weight(model, masks)
