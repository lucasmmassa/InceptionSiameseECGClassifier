from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Flatten, SpatialDropout2D, Lambda
from tensorflow.keras.regularizers import l2

def inception_v1_module(layer_in, f):
    # 1x1 conv
    conv0 = Conv2D(f, (1, 1), padding='same', activation='relu')(layer_in)
    conv1 = Conv2D(f, (1,12), padding='same', activation='relu')(conv0)
    # 3x3 conv
    conv2 = Conv2D(f, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f, (3,12), padding='same', activation='relu')(conv2)
    # 5x5 conv
    conv4 = Conv2D(f, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f, (5,12), padding='same', activation='relu')(conv4)
    # 3x3 max pooling
    pool = MaxPooling2D((3,12), strides=(1,1), padding='same')(layer_in)
    conv6 = Conv2D(f, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, conv6], axis=-1)
    return layer_out

def sister_functional_generation(input_layer):
    inception1 = inception_v1_module(input_layer, 64)
    # drop1 = SpatialDropout2D(0.3)(inception1)
    inception2 = inception_v1_module(inception1, 32)
    # drop2 = SpatialDropout2D(0.3)(inception2)
    pool1 = MaxPooling2D(pool_size=(12, 2))(inception1)
    # inception3 = inception_v1_module(pool1, 64)
    # drop3 = SpatialDropout2D(0.3)(inception3)
    # inception4 = inception_v1_module(inception3, 64)
    # drop4 = SpatialDropout2D(0.3)(inception4)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(inception4)
    # inception5 = inception_v1_module(pool2, 64)
    # drop5 = SpatialDropout2D(0.3)(inception5)
    # inception6 = inception_v1_module(inception5, 64)
    # drop6 = SpatialDropout2D(0.3)(inception6)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(inception6)
    # inception7 = inception_v1_module(pool3, 8, 8, 8)
    # inception8 = inception_v1_module(inception7, 8, 8, 8)
    flat = Flatten()(pool1)
    return flat