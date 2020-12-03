import gluoncv.nn as nn

from gluoncv.nn import ReLU6
from mxnet.gluon import nn,rnn
from mxnet.gluon.nn import BatchNorm
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock


# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False, norm_layer=BatchNorm, norm_kwargs=None):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(norm_layer(scale=True, **({} if norm_kwargs is None else norm_kwargs)))
    if active:
        out.add(ReLU6() if relu6 else nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False,
                 norm_layer=BatchNorm, norm_kwargs=None):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    _add_conv(out, channels=channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=norm_kwargs)


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, in_channels, channels, t, stride,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            if t != 1:
                _add_conv(self.out,
                          in_channels * t,
                          relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      in_channels * t,
                      kernel=3,
                      stride=stride,
                      pad=1,
                      num_group=in_channels * t,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      channels,
                      active=False,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
#         print(f'layer shape{out.shape}')
        return out


def MobileNetV2_small(**kwargs):
    
    return MobileNetV2(multiplier=0.25,**kwargs)

def MobileNetV2_mid(**kwargs):
    return MobileNetV2(multiplier=0.5,**kwargs)

def MobileNetV2_large(**kwargs):
    return MobileNetV2(multiplier=1,**kwargs)

class MobileNetV2(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, multiplier=0.25, classes=41,lstm_hid=256,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(32 * multiplier), kernel=3,
                          stride=2, pad=1, relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [1] + [1] * 3 # [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3 
#                 print(strides)
                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.features.add(LinearBottleneck(in_channels=in_c,
                                                       channels=c,
                                                       t=t,
                                                       stride=s,
                                                       norm_layer=norm_layer,
                                                       norm_kwargs=norm_kwargs))

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features,
                          last_channels,
                          relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            self.lstm = nn.HybridSequential()
            self.lstm.add( 
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                nn.Dense(classes, flatten=False, prefix="pred")
            )

    def hybrid_forward(self, F, x):
        x = self.features(x)
       
        w=x.shape[3]
        x= x.reshape((0, -1, w)).transpose((2, 0, 1))
        out = self.lstm(x).transpose((1, 0, 2))
        return out
 