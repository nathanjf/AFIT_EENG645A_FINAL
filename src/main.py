import os
import numpy as np
from numba import njit

import keras
import tensorflow as tf

from sen12ms_dataTools import SEN12MSDataTools
from sen12ms_sequence import SEN12MSSequence
import modelTools

import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import itertools
import main_config as cfg

figure_base_path = os.path.join(os.path.dirname(__file__), "..", "figures")

model_unet_path = os.path.join(os.path.dirname(__file__), "..", "models", "unet.h5")
model_unet_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "unet")
model_unet_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "unet")

model_characterization_path = os.path.join(os.path.dirname(__file__), "..", "models", "characterization.h5")
model_characterization_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "characterization")
model_characterization_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "characterization")

model_resnet_path = os.path.join(os.path.dirname(__file__), "..", "models", "resnet.h5")
model_resnet_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "resnet")
model_resnet_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "resnet")

def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    this function is from https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)

def get_data(data_ratio : float = 1.0, train_ratio : float = 0.8, val_ratio : float = 0.1, test_ratio : float = 0.1):
    train   : SEN12MSSequence = None
    val     : SEN12MSSequence = None
    test    : SEN12MSSequence = None

    # Get list of data
    sen12ms_datatools : SEN12MSDataTools = SEN12MSDataTools()
    data : np.array = sen12ms_datatools.get_data()
    print(data.shape)
    data = data[0:int(data.shape[0]*data_ratio)]
    print(data.shape)

    # Shuffle the dataset
    np.random.seed(0)
    np.random.shuffle(data)

    # Calculate the start and stop index for all the splits
    train_start_index = 0
    train_stop_index = int(data.shape[0]*train_ratio)
    val_start_index = train_stop_index
    val_stop_index = val_start_index + int(data.shape[0]*val_ratio)
    test_start_index = val_stop_index
    test_stop_index = test_start_index + int(data.shape[0]*test_ratio)

    # Split the index file
    train_data  = data[train_start_index:train_stop_index,  :]
    val_data    = data[val_start_index:val_stop_index,      :]
    test_data   = data[test_start_index:,                   :]

    # Make sure all splits add up to the expected dataset size
    # print(train_data.shape)
    # print(val_data.shape)
    # print(test_data.shape)
    # print(train_data.shape[0] + val_data.shape[0]+ test_data.shape[0])

    # Initialize the sequences
    train = SEN12MSSequence(train_data, cfg.BATCH_SIZE)
    val = SEN12MSSequence(val_data, cfg.BATCH_SIZE)
    test = SEN12MSSequence(test_data, cfg.BATCH_SIZE)

    return train, val, test

def main():
    # Configure workspace
    modelTools.set_gpu_gemory_growth()

    # Data Prep Phase

    # Step 1
    # Get data

    # Step 2
    # Measures of success
    # Categorical accuracy

    # Step 3
    # Prepare data

    train, val, test = get_data(0.5)

    # Training Phase
    model : keras.models.Model = None

    match cfg.MODEL:
        case cfg.Model.CHARACTERIZATION:

            model_path = model_characterization_path
            model_figure_path = model_characterization_figure_path
            model_log_path = model_characterization_log_path

        case cfg.Model.UNET:

            model_path = model_unet_path
            model_figure_path = model_unet_figure_path
            model_log_path = model_unet_log_path

        case cfg.Model.RESNET:
            
            model_path = model_resnet_path
            model_figure_path = model_resnet_figure_path
            model_log_path = model_resnet_log_path

    # Do model training or load model
    if cfg.TRAIN:
        
        # Step 5
        # Baseline Model
        match cfg.MODEL:
            case cfg.Model.CHARACTERIZATION:

                model = modelTools.characterization(
                    input_shape=(256,256,15),
                    num_classes=10,
                    filters=8, 
                    kernel_size=3
                )

            case cfg.Model.UNET:

                model = modelTools.unet(
                    input_shape=(256,256,15),
                    num_classes=10,
                    filters=8, 
                    kernel_size=3
                )

            case cfg.Model.RESNET:

                model = modelTools.resnet(
                    input_shape=(256,256,15),
                    num_classes=10,
                    filters=64, 
                    kernel_size=3
                )

        # Plot the model
        keras.utils.plot_model(model, to_file=os.path.join(model_figure_path, 'model.png'), show_shapes=True, show_layer_names=False, show_layer_activations=True)

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.00001
            ),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        train.calculate_class_weights()
        class_weights = train.get_class_weights()

        # Fit the model
        model.fit(
            x=train,
            validation_data=val,
            class_weight=class_weights,
            batch_size=cfg.BATCH_SIZE,
            epochs=cfg.EPOCHS,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3), 
                keras.callbacks.TensorBoard(log_dir=model_log_path, update_freq='epoch'), 
                modelTools.ClearMemory()
            ],
            workers=cfg.WORKERS,
            use_multiprocessing=cfg.MULTIPROCESSING
        )

        model.save(model_path)
    else:
        model = keras.models.load_model(model_path)
        pass

    # Choose prediction set
    x_set : SEN12MSSequence = None
    do_test : bool = True
    if do_test:
        x_set = test
    else:
        x_set = val

    # Plot some band combinations
    plot_x, _ = x_set.get_item(3)
    print(plot_x.shape)
    SEN12MSDataTools.plot_bands(plot_x, figure_base_path)   

    y_true = np.zeros([x_set.data.shape[0], 256, 256])
    y_pred = np.zeros([x_set.data.shape[0], 256, 256])
    
    for pred_idx in range(0, x_set.data.shape[0]):
        print("predicting", pred_idx, "of", x_set.data.shape[0])
        x, y = x_set.get_item(pred_idx)
        
        y_true[pred_idx] = np.argmax(y, axis=3)
        y_pred[pred_idx] = (np.argmax(model(np.reshape(x.copy(), (1,256,256,15))), axis=3))
        
        y = np.reshape(np.argmax(y, axis=3) + 1, (256,256))
        y_p = np.reshape(np.argmax(model(np.reshape(x.copy(), (1,256,256,15))), axis=3) + 1, (256,256))     
        if pred_idx < 128:   
            SEN12MSDataTools.plot_prediction(x, y, y_p, os.path.join(model_figure_path, f"prediction_{pred_idx}.png"))


    # Transform into an array of pixelwise predictions after the fact
    y_true = np.reshape(y_true, [y_true.shape[0] * y_true.shape[1] * y_true.shape[2]])
    y_pred = np.reshape(y_pred, [y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2]])
    
    # Print just accuracy.  Classification report is too slow.  So slow that I'm 99% sure there's a bug in it
    # that the scikit-learn team hasn't run into.  Doing a little bit of searching it seems like there's unneccesary calls to a full
    # sort of the arrays.
    print(accuracy_score(y_true, y_pred, normalize=True))

    # calculate the confusion matrix
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')

    # plot the confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, cfg.simple_just_names, True, os.path.join(model_figure_path, 'confusion_matrix'))

if __name__ == "__main__":
    main()