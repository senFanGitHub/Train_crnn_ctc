import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import Block
from mxnet.gluon import nn, rnn
from mxnet import nd
from mxnet.gluon.contrib.nn import Concurrent, Identity



def _make_dense_layer(growth_rate, dropout):
    new_features = nn.Sequential(prefix='')
    new_features.add(nn.BatchNorm())
    new_features.add(nn.Activation('relu'))
    new_features.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
    if dropout:
        new_features.add(nn.Dropout(dropout))

    out = Concurrent(axis=1, prefix='')
    out.add(Identity())
    out.add(new_features)
    return out



def _make_dense_block(nb_layers, growth_rate, dropout, stage_index):
    out = nn.Sequential(prefix='stage%d_'%stage_index)
    with out.name_scope():
        for _ in range(nb_layers):
            out.add(_make_dense_layer(growth_rate, dropout))

            
    return out 

def _make_transition(num_output_features):
    out = nn.Sequential(prefix='')
    out.add(nn.BatchNorm())
    out.add(nn.Activation('relu'))
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    out.add(nn.AvgPool2D(pool_size=2, strides=2))
    return out


class DenseNet(Block):

    def __init__(self, num_init_features=64, growth_rate=32, dropout=0.,
                 classes=41,lstm_hid=256, **kwargs):#bn_size=4, 

        super(DenseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential(prefix='')
            self.features.add(nn.Conv2D(num_init_features, kernel_size=5,
                                        strides=2, padding=2, use_bias=False))
            num_features = num_init_features
            self.features.add(_make_dense_block(8, growth_rate, dropout,1))
            num_features = num_features + 8 * growth_rate
            self.features.add(_make_transition(128) )
            
            self.features.add(_make_dense_block(16, growth_rate, dropout,2))
            num_features = num_features + 16 * growth_rate
            self.features.add(_make_transition(128) )
            
            self.features.add(_make_dense_block(24, growth_rate, dropout,3))
            num_features = num_features + 24 * growth_rate
            self.features.add(_make_transition(128) )
            
            self.features.add(_make_dense_block(32, growth_rate, dropout,4))
            num_features = num_features + 32 * growth_rate
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            
            self.lstm = nn.Sequential()
            self.lstm.add( 
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                rnn.LSTM(lstm_hid, dropout=0.2, bidirectional=True),
                nn.Dense(classes, flatten=False, prefix="pred")
            )
            

    def forward(self, x):
        x = self.features(x)
        w=x.shape[3]
        x= x.reshape((0, -1, w)).transpose((2, 0, 1))
        out = self.lstm(x).transpose((1, 0, 2))
        return out