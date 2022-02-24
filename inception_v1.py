from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Flatten, SpatialDropout2D, Lambda
from tensorflow.keras.regularizers import l2

def inception_v1_module(layer_in, f_in, f_out):
    # 1x1 conv
    conv0_0 = Conv2D(f_out, (1, 1), padding='same', activation='relu')(layer_in)

    conv1_0 = Conv2D(f_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv1 = Conv2D(f_out, (7,1), padding='same', activation='relu')(conv1_0)
    conv1_1 = Conv2D(f_out, (1,7), padding='same', activation='relu')(conv1)
    # 3x3 conv
    # conv2 = Conv2D(f, (1, 1), padding='same', activation='tanh')(layer_in)
    conv3_0 = Conv2D(f_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f_out, (9,1), padding='same', activation='relu')(conv3_0)
    conv3_1 = Conv2D(f_out, (1,9), padding='same', activation='relu')(conv3)
    # 5x5 conv
    # conv4 = Conv2D(f, (1, 1), padding='same', activation='tanh')(layer_in)
    conv5_0 = Conv2D(f_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f_out, (12,1), padding='same', activation='relu')(conv5_0)
    conv5_1 = Conv2D(f_out, (1,12), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    conv_pool = Conv2D(f_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv0_0, conv1_1, conv3_1, conv5_1, conv_pool], axis=-1)
    return layer_out

def sister_functional_generation(input_layer):
    inception1 = inception_v1_module(input_layer, 16, 16)
    inception2 = inception_v1_module(inception1, 8, 9)
    flat = Flatten()(inception2)
    return flat