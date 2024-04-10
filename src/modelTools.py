import gc
import keras
import tensorflow as tf
import math
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPooling2D, Add, Concatenate, UpSampling2D

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

def unet(input_shape=None, num_classes=None, filters=32, kernel_size=3):
    """
        Basic UNET
    """

    inputs : keras.layers.Layer
    outputs : keras.layers.Layer

    # Determine how many modules are needed to get to a 1x1 convolution
    max_depth = int(math.log(input_shape[0], 2))

    # Inputs
    inputs = Input(shape=input_shape)

    # Encode
    skips = []
    conv = inputs
    for depth in range(0, max_depth):
        # Conv block
        conv = Conv2D(filters=filters*(2**(depth)), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
        conv = Conv2D(filters=filters*(2**(depth)), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=2)(batch)
        
        # Add conv to the skip array
        skips.append(conv)
        
        conv = pool

    # Core convolution blocks
    conv = Conv2D(filters=filters*(2**max_depth), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
    conv = Conv2D(filters=filters*(2**max_depth), kernel_size=kernel_size, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(conv)

    # Decode
    for depth in range(0, max_depth):
        deconv = Conv2DTranspose(filters=filters*(2**(max_depth-depth)), kernel_size=kernel_size, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
        concat = Concatenate(axis=3)([deconv, skips[max_depth-depth-1]])
        
        conv = concat

    # Predict
    outputs = Conv2D(filters=num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax', kernel_initializer='he_normal')(conv)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model

def resnet_layer(input_layer, filters, kernel_size):
    
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(input_layer)
    batch = BatchNormalization()(conv)
    relu = ReLU()(batch)    
    
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(relu)
    batch = BatchNormalization()(conv)
    relu = ReLU()(batch)

    add = Add()([input_layer, relu])
    relu = ReLU()(add)

    return relu

def resnet(input_shape=None, num_classes=None, filters=32, kernel_size=3):
    inputs : keras.layers.Layer
    outputs : keras.layers.Layer

    inputs = Input(shape=input_shape)

    max_depth = int(math.log(input_shape[0], 2))
    # Prepare data for resnet
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(inputs)
    batch = BatchNormalization()(conv)    

    # Resnet
    previous_layer = batch
    for i in range(0, max_depth):
        next_layer = resnet_layer(previous_layer, filters=filters, kernel_size=kernel_size)
        previous_layer = next_layer
    
    # Prediction
    outputs = Conv2D(filters=num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax')(previous_layer)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model