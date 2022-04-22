from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Input, Reshape, Activation, Multiply
from kears.models import Model
import math

def eca_block(inputs, b=1, gamma=2):
    # inputs: h, w, c
    channel = inputs.keras.shape[-1]
    # x: c
    x = GlobalAveragePooling2D()(inputs)
    # x: 1, 1, c
    x = Reshape([1, 1, -1])(x)

    # x = Dense(channel // 4)(x)
    # x = Activation('relu')(x)
    # x = Dense(channel)(x)
    # x = Activation('sigmoid')(x)

    x = Conv2D(channel // 4, [1, 1])(x)
    x = Activation('relu')(x)
    x = Conv2D(channel, [1, 1])(x)
    x = Activation('sigmoid')(x)

    out = Multiply()([x, inputs])
    return out

inputs = Input([26, 26, 512]) 
x = se_block(inputs)
model = Model(inputs, x)
model.summary()