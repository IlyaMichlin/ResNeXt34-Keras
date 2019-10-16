# Introduction

Implementattion of the ResNeXt 34 architecture [1] and conversion of pre-trained weights on imagenet from PyTorch format to Keras format.

## How to use

To generate model with imagenet weights use the following code:
n_classes: number of desired output classes
weights: if 'imagenet' then loads imagenet weights without the last layed and attaches a new Dense layer with n_classes outputs (expects filenamed 'resnet34-333f7ec4.pth' [2]). else will generate random weights
returns ResNeXt 34 model

```sh
from ResNeXt34 import resnext34

model = resnext34(input_shape, n_classes=1000, weights='imagenet')
```

To load your own weights, use the following code:

```sh
from ResNeXt34 import load_weights

model = load_weights(model, file_path)
```

Running ResNeXt34.py will generate and save Keras model and weights. Expects filenamed 'resnet34-333f7ec4.pth' the can be downloaded from link [2].
Also, there is an example of saving and loading the model and the weights so use that.

### Links

[1] Model Architecture:
* https://github.com/fastai/imagenet-fast/blob/master/cifar10/models/resnext.py

[2] Download PyTorch weights:
* https://download.pytorch.org/models/resnet34-333f7ec4.pth