from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Flatten, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2

def inception_v1_module(layer_in, f_in, f_out):
    # 1x1 conv
    conv0 = Conv2D(f_out, (1, 1), padding='same')(layer_in)
    # conv0 = BatchNormalization()(conv0)

    # conv1 = Conv2D(f_in, (1, 1), padding='same')(layer_in)
    conv1 = Conv2D(f_out, (3,1), padding='same')(layer_in)
    conv1 = Conv2D(f_out, (1,3), padding='same')(conv1)
    
    # conv2 = Conv2D(f_in, (1, 1), padding='same')(layer_in)
    conv2 = Conv2D(f_out, (5,1), padding='same')(layer_in)
    conv2 = Conv2D(f_out, (1,5), padding='same')(conv2)
    
    # conv3 = Conv2D(f_in, (1, 1), padding='same')(layer_in)
    conv3 = Conv2D(f_out, (7,1), padding='same')(layer_in)
    conv3 = Conv2D(f_out, (1,7), padding='same')(conv3)
    
    # conv4 = Conv2D(f_in, (1, 1), padding='same')(layer_in)
    # conv4 = Conv2D(f_out, (9,1), padding='same')(layer_in)
    # conv4 = Conv2D(f_out, (1,9), padding='same')(conv4)
    
    # # conv5 = Conv2D(f_in, (1, 1), padding='same')(layer_in)
    # conv5 = Conv2D(f_out, (12,1), padding='same')(layer_in)
    # conv5 = Conv2D(f_out, (1,12), padding='same')(conv5)

    # pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # conv_pool = Conv2D(f_out, (1, 1), padding='same')(pool)
    
    layer_out = concatenate([conv0, conv1, conv2, conv3], axis=-1)
    layer_out = ReLU(layer_out)
    return layer_out

def sister_functional_generation(layer):
    layer = inception_v1_module(layer, 64, 32)
    layer = inception_v1_module(layer, 32, 16)
    layer = inception_v1_module(layer, 16, 8)
    layer = inception_v1_module(layer, 8, 8)
    layer = Flatten()(layer)
    # layer = Dense(32)(layer)
    # layer = BatchNormalization()(layer)
    # layer = ReLU()(layer)
    # layer = Dense(16)(layer)
    # layer = BatchNormalization()(layer)
    # layer = ReLU()(layer)
    return layer