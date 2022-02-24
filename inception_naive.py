from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Flatten, SpatialDropout2D, Lambda

def inception_naive_module(layer_in,f):
    # 1x1 conv
    conv1 = Conv2D(f, (1,1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f, (3,3), padding='same', activation='relu')(layer_in)
    # 5x5 conv
    conv5 = Conv2D(f, (5,5), padding='same', activation='relu')(layer_in)
    # 3x3 max pooling
    # pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5], axis=-1)
    return layer_out

def sister_functional_generation(input_layer):
    inception1 = inception_naive_module(input_layer, 128)
    inception2 = inception_naive_module(inception1, 128)
    # drop1 = SpatialDropout2D(0.2)(inception2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(inception2)
    inception3 = inception_naive_module(pool1, 64)
    inception4 = inception_naive_module(inception3, 64)
    # drop2 = SpatialDropout2D(0.2)(inception4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(inception4)
    inception5 = inception_naive_module(pool2, 64)
    inception6 = inception_naive_module(inception5, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(inception6)
    # inception7 = inception_module(pool3, 32, 32, 32)
    # inception8 = inception_module(inception7, 32, 32, 32)
    flat = Flatten()(pool3)
    return flat