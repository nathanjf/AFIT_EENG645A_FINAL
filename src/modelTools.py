import gc
import keras
import tensorflow as tf
import math
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPooling2D, Add, Concatenate

class ClearMemory(keras.callbacks.Callback):
    """
    Custom callback that manually runs garbage collection at the end of each epoch to avoid a memory leak present in tensorflow
    """
    def on_epoch_end(self, epoch, logs=None):
        keras.backend.clear_session()
        gc.collect()

def set_gpu_gemory_growth():
    """
    Set memory growth for all gpus
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def RESNetLayer(input_layer, filters, kernel_size):
    
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(input_layer)
    batch = BatchNormalization()(conv)
    relu = ReLU()(batch)    
    
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(relu)
    batch = BatchNormalization()(conv)
    relu = ReLU()(batch)

    add = Add()([input_layer, relu])
    relu = ReLU()(add)

    return relu

def model_1(input_shape=None, num_classes=None, resnet_depth=50, resnet_filters=16, kernel_size=3):
    inputs : keras.layers.Layer
    outputs : keras.layers.Layer

    inputs = Input(shape=input_shape)

    # Prepare data for resnet
    conv = Conv2D(filters=resnet_filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(inputs)
    batch = BatchNormalization()(conv)    

    # Resnet
    previous_layer = batch
    for i in range(0, resnet_depth):
        next_layer = RESNetLayer(previous_layer, filters=resnet_filters, kernel_size=kernel_size)
        previous_layer = next_layer
    
    # Prediction
    outputs = Conv2D(filters=num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax')(previous_layer)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model

def WideUNetLayer(input_layer, width, filters, kernel_size):
    paths = []
    for i in range(0, width):
        # Downsample up to i times
        skips = []
        conv = input_layer
        for depth in range(0, i):
            # Conv block
            conv = Conv2D(filters=filters*(2**(depth)), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
            conv = Conv2D(filters=filters*(2**(depth)), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
            pool = MaxPooling2D(pool_size=2)(conv)
            
            # Add conv to the skip array
            skips.append(conv)
            
            conv = pool

        # Core convolution blocks
        conv = Conv2D(filters=filters*width, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
        conv = Conv2D(filters=filters*width, kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)

        # Upsample up to i times
        for depth in range(0, i):
            deconv = Conv2DTranspose(filters=filters*(2**(i-depth)), kernel_size=kernel_size, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
            concat = Concatenate(axis=3)([deconv, skips[i-depth-1]])
            
            conv = concat

        paths.append(conv)

    concat = Concatenate()(paths)

    return concat

def model_2(input_shape=None, num_classes=None, filters=32, kernel_size=3):
    inputs : keras.layers.Layer
    outputs : keras.layers.Layer

    # Determine how many unets are needed to get to a 1x1 convolution
    width = int(math.log(input_shape[0], 2))

    # Inputs
    inputs = Input(shape=input_shape)

    # Resnext
    wide_unet = WideUNetLayer(inputs, width=width, filters=filters, kernel_size=kernel_size)
    
    # Prediction
    outputs = Conv2D(filters=num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax', kernel_initializer='he_normal')(wide_unet)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model


def model_3(input_shape=None, num_classes=None, filters=32, kernel_size=3):
    
    # TODO : Standard Unet
    
    pass
