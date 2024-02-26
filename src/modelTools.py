import gc
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, MaxPooling2D, Add

class ClearMemory(keras.callbacks.Callback):
    """
    Custom callback that manually runs garbage collection at the end of each epoch to avoid a memory leak present in tensorflow
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()

def set_gpu_gemory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
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

    # Resnet50
    previous_layer = batch
    for i in range(0,resnet_depth):
        next_layer = RESNetLayer(previous_layer, filters=resnet_filters, kernel_size=kernel_size)
        previous_layer = next_layer
    
    # Encoder
        
    # Decoder

    # Prediction
    outputs = Conv2D(filters=num_classes, kernel_size=kernel_size, strides=1, padding='same', activation='softmax')(previous_layer)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model

def model_old():
    
    input : keras.layers.Layer
    output : keras.layers.Layer

    input = Input(shape=(256,256,13))

    # CNN
    conv_layer_1    = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(input)
    conv_layer_2    = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(conv_layer_1)
    batch_norm_1    = BatchNormalization()(conv_layer_2)
    pooling_layer_1 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    
    conv_layer_3    = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', activation='relu')(pooling_layer_1)
    conv_layer_4    = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', activation='relu')(conv_layer_3)
    batch_norm_2    = BatchNormalization()(conv_layer_4)
    pooling_layer_2 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(batch_norm_2)

    conv_layer_5    = Conv2D(filters=512, kernel_size=2, strides=1, padding='same', activation='relu')(pooling_layer_2)
    conv_layer_6    = Conv2D(filters=512, kernel_size=2, strides=1, padding='same', activation='relu')(conv_layer_5)
    batch_norm_3    = BatchNormalization()(conv_layer_6)
    pooling_layer_3 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(batch_norm_3)

    conv_layer_7    = Conv2D(filters=1024, kernel_size=2, strides=1, padding='same', activation='relu')(pooling_layer_3)
    conv_layer_8    = Conv2D(filters=1024, kernel_size=2, strides=1, padding='same', activation='relu')(conv_layer_7)
    batch_norm_4    = BatchNormalization()(conv_layer_8)
    pooling_layer_4 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(batch_norm_4)

    # Passthroughs
    # TODO : Upsample passthrough layers to match intermediate steps
    passthrough_1 = batch_norm_1
    passthrough_2 = batch_norm_2
    passthrough_3 = batch_norm_3
    passthrough_4 = batch_norm_4

    # FNN

    #upsampling_layer_1 = UpSampling2D(size=(4,4))(pooling_layer_4)
    deconv_layer_1 = Conv2DTranspose(filters=1024, kernel_size=2, strides=2, padding='same', activation='relu')(pooling_layer_4)
    conv_layer_9    = Conv2D(filters=1024, kernel_size=2, strides=1, padding='same', activation='relu')(deconv_layer_1)
    #add_layer_1 = Add()([conv_layer_9, passthrough_4])

    #upsampling_layer_2 = UpSampling2D(size=(4,4))(add_layer_1)
    deconv_layer_2 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='same', activation='relu')(conv_layer_9)
    conv_layer_10    = Conv2D(filters=512, kernel_size=2, strides=1, padding='same', activation='relu')(deconv_layer_2)
    add_layer_2 = Add()([conv_layer_10, passthrough_3])

    #upsampling_layer_3 = UpSampling2D(size=(4,4))(add_layer_2)
    deconv_layer_3 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='same', activation='relu')(add_layer_2)
    conv_layer_11    = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', activation='relu')(deconv_layer_3)
    add_layer_3 = Add()([conv_layer_11, passthrough_2])

    #upsampling_layer_4 = UpSampling2D(size=(4,4))(add_layer_3)
    deconv_layer_4 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='same', activation='relu')(add_layer_3)
    conv_layer_12    = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(deconv_layer_4)
    add_layer_4 = Add()([conv_layer_12, passthrough_1])

    output = Conv2DTranspose(filters=17, kernel_size=2, strides=1, padding='same', activation='softmax')(add_layer_4)

    model = keras.models.Model(inputs=input, outputs=output)
    
    pass