from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, GlobalMaxPool2D, Input, Reshape, Activation, Multiply, Add
from kears.models import Model
from keras.layers.core import lambda
from keras import backend as

def channel_block(inputs, ratio=4):
    # inputs: h, w, c
    channel = inputs.keras.shape[-1]
    
    share_dense1 = Dense(channel // ratio, activation='relu')
    share_dense2 = Dense(channel)

    # x: c    
    x_max = GlobalMaxPool2D()(inputs)
    x_avg = GlobalAveragePooling2D()(inputs)


    # x: 1, 1, c
    x_max = Reshape([1, 1, -1])(x_max)
    x_avg = Reshape([1, 1, -1])(x_avg)

    x_max = share_dense1()(x_max)
    x_max = share_dense2()(x_max)

    x_avg = share_dense1()(x_avg)
    x_avg = share_dense2()(x_avg)

    x = Add()([x_max, x_avg])
    x = Activation('sigmoid')(x)

    # x = Conv2D(channel // 4, [1, 1])(x)
    # x = Activation('relu')(x)
    # x = Conv2D(channel, [1, 1])(x)
    # x = Activation('sigmoid')(x)

    out = Multiply()([x, inputs])
    return out

def spatial_block(inputs, ratio=4):
    # inputs: h, w, c
    channel = inputs.keras.shape[-1]
    
    share_dense1 = Dense(channel // ratio, activation='relu')
    share_dense2 = Dense(channel)

    # x: c    
    x_max = GlobalMaxPool2D()(inputs)
    x_avg = GlobalAveragePooling2D()(inputs)


    # x: 1, 1, c
    x_max = Reshape([1, 1, -1])(x_max)
    x_avg = Reshape([1, 1, -1])(x_avg)

    x_max = share_dense1()(x_max)
    x_max = share_dense2()(x_max)

    x_avg = share_dense1()(x_avg)
    x_avg = share_dense2()(x_avg)

    x = Add()([x_max, x_avg])
    x = Activation('sigmoid')(x)

    # x = Conv2D(channel // 4, [1, 1])(x)
    # x = Activation('relu')(x)
    # x = Conv2D(channel, [1, 1])(x)
    # x = Activation('sigmoid')(x)

    out = Multiply()([x, inputs])
    return out

inputs = Input([26, 26, 512]) 
x = channel_block(inputs)
model = Model(inputs, x)
model.summary()