
from sen12ms_sequence import SEN12MSSequence

import os
import keras
import tensorflow as tf
import numpy as np

import tensorflow as tf

import gc
from keras import backend as k
from keras.layers import Conv2D, BatchNormalization, ReLU
from keras.callbacks import Callback

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Add, BatchNormalization

from keras.utils import plot_model

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()
        print("Cleared memory")

def main():

    batch_size = 4
    epochs = 100

    # Step 1
    # Get data


    # Step 2
    # Measures of success
    # Categorical accuracy

    # Step 3
    # Prepare data

    sen12ms = SEN12MSSequence(batch_size=batch_size)
    

    # Step 4
    # Evaluation Method

    # TODO : Extract data list
    # Split data list into training, testing, and validation splits
    # Create training, testing, and validation data splits
    # TODO : Add the ability to split the generator into different sets by feeding it a predefined list of indexes

    # Calculate weights
    # TODO : Only calculate on the training set
    counts = {
            1: 35169562, 
            2: 114461610, 
            3: 8516, 
            4: 173018642, 
            5: 91931871, 
            6: 15266338, 
            7: 59174553, 
            8: 177388877, 
            9: 518445158, 
            10: 439273502, 
            11: 31064884, 
            12: 507802578, 
            13: 335295405, 
            14: 42061406, 
            15: 37774, 
            16: 1904431, 
            17: 137003181
    }
    weights = {}
    total = 0
    for key in counts.keys():
        total += counts[key]
    for key in counts.keys():
        weights[key-1] = float(total) / (float(len(counts.keys())) * float(counts[key]))

    model : keras.models.Model = None
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.h5")

    do_train_model : bool = True
    if do_train_model or not os.path.exists(model_path):

        # Step 5
        # Baseline Model

        do_first_fit : bool = False
        if do_first_fit:

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

            model.summary()
            plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
            pass

        # Step 6
        # Overfit Model

        do_second_fit : bool = False
        if do_second_fit:
        
            # TODO : 

            pass

        # Step 7
        # Regularize Model

        do_third_fit : bool = False
        if do_third_fit:
            
            # TODO :
            
            pass

        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
            #run_eagerly=True
        )

        model.fit(
            x=sen12ms,
            batch_size=batch_size,
            epochs=epochs,
            workers=4,
            use_multiprocessing=True,
            class_weight=weights,
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3), ClearMemory()]
        )

        model.save(model_path)
    else:
        model = keras.models.load_model(model_path)
        pass

    # Test prediction by grabbing a random element and plotting it
    #sen12ms.plot_item(10)
    x, y = sen12ms.__getitem__(10)
    # TODO : Swap y to argmax also
    y_pred = np.argmax(model.predict(x), axis=3, keepdims=True)

    sen12ms.plot_predictions(x=x,y=y,y_pred=y_pred,idx=3)

    # Step 8
    # Test Set Performance

if __name__ == "__main__":
    main()