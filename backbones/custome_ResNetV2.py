
"""ResNets, implemented in Gluon."""
from __future__ import division


from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn,rnn
from mxnet.gluon.nn import BatchNorm

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)



class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = _conv3x3(channels, stride, in_channels)
        if not last_gamma:
            self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn2 = norm_layer(gamma_initializer='zeros',
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels, 1, channels)

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 16, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = _conv3x3(channels//4, stride, channels//4)
        if not last_gamma:
            self.bn3 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn3 = norm_layer(gamma_initializer='zeros',
                                  **({} if norm_kwargs is None else norm_kwargs))
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 16, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        return x + residual



class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, block, layers, channels, classes=41,lstm_hid=256,thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False,
                                         **({} if norm_kwargs is None else norm_kwargs)))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels,
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                in_channels = channels[i+1]
            self.features.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.features.add(nn.Activation('relu'))


            self.lstm = nn.HybridSequential()
            self.lstm.add( 
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                nn.Dense(classes, flatten=False, prefix="pred")
            )
            
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, prefix='',
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        w=x.shape[3]
        x= x.reshape((0, -1, w)).transpose((2, 0, 1))
        out = self.lstm(x).transpose((1, 0, 2))
        return out

# # Specification
# resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
#                34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
#                50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
#                101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
#                152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

# resnet_net_versions = [ResNetV1, ResNetV2]
# resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
#                          {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]

def ResNetV2_small(**kwargs):
    return ResNetV2(BasicBlockV2,[2, 2, 2], [64, 64, 128, 256],**kwargs)
def ResNetV2_mid(**kwargs):
    return ResNetV2(BasicBlockV2,[3, 4, 6], [64, 64, 128, 256],**kwargs)
def ResNetV2_large(**kwargs):
    return ResNetV2(BottleneckV2,[3, 4, 6], [64, 256, 512, 1024],**kwargs)
