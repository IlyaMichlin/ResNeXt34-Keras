import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Add, Dense, GlobalAveragePooling2D


def load_weights(model, file_path, include_top=False, is_torch=False):
    """loads and sets saved weights to the Keras model

    :param model: Keras model
    :param file_path: path to the weights file
    :param include_top: indicates if include the top weights
    :param is_torch: indicates if it is a PyTorch file
    :return: model with loaded weights
    """

    if is_torch:
        import torch

        pretrain_state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        # layers = list(pretrain_state_dict.keys())
        layers = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.conv2.weight', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.1.conv1.weight', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.conv2.weight', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.2.conv1.weight', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias', 'layer1.2.conv2.weight', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.conv2.weight', 'layer2.0.downsample.0.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.2.conv1.weight', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.conv2.weight', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.3.conv1.weight', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.conv2.weight', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.conv2.weight', 'layer3.0.downsample.0.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.2.conv1.weight', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.conv2.weight', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.3.conv1.weight', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias', 'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.conv2.weight', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.4.conv1.weight', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.conv2.weight', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.5.conv1.weight', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.conv2.weight', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.conv2.weight', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'fc.weight', 'fc.bias']
        if not include_top:
            layers = layers[:-2]

        weights = []
        for layer in layers:
            weights.append(np.transpose(pretrain_state_dict[layer].detach().numpy()))

        if not include_top:
            old_weights = model.get_weights()

            for n in range(len(old_weights) - len(weights), 0, -1):
                weights.append(old_weights[-n])

        model.set_weights(weights)
    else:
        model.load_weights(file_path)

    return model


def input_layer(inputs, filters_n, name=''):
    x = Conv2D(filters_n, kernel_size=7, strides=2, padding='same', use_bias=False, name=name+'_conv2d')(inputs)
    x = BatchNormalization(name=name+'_bn')(x)
    x = ReLU(name=name+'_relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name=name+'_pool')(x)

    return x


def ConvBn2D(x, filters_n, kernel_size=3, strides=1, padding='same', name=''):
    x = Conv2D(filters_n, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=name+'_conv2d')(x)
    x = BatchNormalization(epsilon=1e-5, name=name+'_bn')(x)

    return x


def BasicBlock(x, filters_n, strides=1, is_shortcut=False, name=''):
    x_new = ConvBn2D(x, filters_n, kernel_size=3, strides=strides, padding='same', name=name+'_0')
    x_new = ReLU(name=name+'_0_relu')(x_new)
    x_new = ConvBn2D(x_new, filters_n, kernel_size=3, strides=1, padding='same', name=name+'_1')

    if is_shortcut:
        x_skip = ConvBn2D(x, filters_n, kernel_size=1, strides=strides, padding='same', name=name+'_2')
        x_new = Add(name=name+'_2_add')([x_new, x_skip])

    x_new = ReLU(name=name+'_2_relu')(x_new)

    return x_new


def _make_layer(x, filters_n, layers, strides=1, is_shortcut=False, name=''):
    x = BasicBlock(x, filters_n, strides=strides, is_shortcut=is_shortcut, name=name+'_bb0')
    for layer in range(1, layers):
        x = BasicBlock(x, filters_n, strides=1, is_shortcut=False, name=name+'_bb'+str(layer))

    return x


def resnext34(input_shape, n_classes=1000, weights='imagenet'):
    """build ResNext 34 model

    https://github.com/fastai/imagenet-fast/blob/master/cifar10/models/resnext.py

    :param input_shape: model input shape
    :param n_classes: number of output classes
    :param weights: initialization weights
    :return: ResNeXt 34 model
    """

    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]

    layers = [3, 4, 6, 3]
    filters_n = [64, 128, 256, 512]

    # inputs
    inputs = Input(input_shape)

    # inner layers
    layer0 = input_layer(inputs, filters_n[0], name='layer0')
    layer1 = _make_layer(layer0, filters_n[0], layers[0], strides=1, is_shortcut=False, name='layer1')
    layer2 = _make_layer(layer1, filters_n[1], layers[1], strides=2, is_shortcut=True, name='layer2')
    layer3 = _make_layer(layer2, filters_n[2], layers[2], strides=2, is_shortcut=True, name='layer3')
    layer4 = _make_layer(layer3, filters_n[3], layers[3], strides=2, is_shortcut=True, name='layer4')

    # output layer
    avg_pool = GlobalAveragePooling2D(name='layer5_pool')(layer4)
    outputs = Dense(n_classes, name='layer5_dense')(avg_pool)

    # build model
    model = Model(inputs=inputs, outputs=outputs)

    if weights == 'imagenet':
        file_path = 'resnet34-333f7ec4.pth'
        model = load_weights(model, file_path, include_top=False, is_torch=True)

    return model


if __name__ == '__main__':
    # configurations
    input_shape = (256,1600,3)

    # generate model
    print('Generating ResNeXt 34 model')
    model = resnext34(input_shape)

    # print(model.summary())

    weights = np.array(model.get_weights())
    print('Number of layers: {}'.format(weights.shape))

    # for n in range(weights.shape[0]):
    #     # print('layer {} shape: {}'.format(n, weights[n].shape))
    #     print(weights[n].shape)
    #     # print(weights[n])

    # generate imagenet weights
    print('Loadding imagenet weights')
    file_path = 'resnet34-333f7ec4.pth'
    model = load_weights(model, file_path, include_top=False, is_torch=True)

    # save model and weights
    print('Saving model')
    model_json = model.to_json()
    with open("resnext34.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("resnext34.h5")
    print("Saved model to disk")

    # load model and weights
    print('Loading model')
    from tensorflow.keras.models import model_from_json
    json_file = open('resnext34.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("resnext34.h5")
    print("Loaded model from disk")
