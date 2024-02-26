import os
import numpy as np

import keras
import tensorflow as tf

from sen12ms_dataTools import SEN12MSDataTools
from sen12ms_sequence import SEN12MSSequence
import modelTools

EPOCHS = 100
BATCH_SIZE = 4

figure_base_path = os.path.join(os.path.dirname(__file__), "..", "figures")

model_1_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_1.keras")
model_1_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "model_1.png")
model_1_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "model_1")
model_2_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_2.keras")
model_2_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "model_2.png")
model_2_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "model_2")
model_3_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_3.keras")
model_3_figure_path = os.path.join(os.path.dirname(__file__), "..", "figures", "model_3.png")
model_3_log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "model_3")

def get_data(data_ratio : float = 1.0, train_ratio : float = 0.8, val_ratio : float = 0.1, test_ratio : float = 0.1):
    train   : SEN12MSSequence = None
    val     : SEN12MSSequence = None
    test    : SEN12MSSequence = None

    # Get list of data
    sen12ms_datatools : SEN12MSDataTools = SEN12MSDataTools()
    data : np.array = sen12ms_datatools.get_data()
    data = data[0:int(data.shape[0]*data_ratio)]

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
    test_data   = data[test_start_index:,    :]

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)
    print(train_data.shape[0] + val_data.shape[0]+ test_data.shape[0])

    # Initialize the sequences
    train = SEN12MSSequence(train_data, BATCH_SIZE)
    val = SEN12MSSequence(val_data, BATCH_SIZE)
    test = SEN12MSSequence(test_data, BATCH_SIZE)

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

    train, val, test = get_data(0.05) # TODO : IN THE FINAL FIT MAKE SURE THIS IS SET TO 1
    train.calculate_class_weights()
    class_weights = train.get_class_weights()
    
    # Training Phase

    model : keras.models.Model = None

    # Chose which model
    do_train_model : bool = True
    do_first_model : bool = True
    do_second_model : bool = False
    do_third_model : bool = False
    if not (do_first_model ^ do_second_model ^ do_third_model):
        raise RuntimeError("Only one model can be true at at time")
    
    # Set file path for selected model
    model_path = None
    model_figure_path = None
    if do_first_model:        
        model_path = model_1_path
        model_figure_path = model_1_figure_path
        model_log_path = model_1_log_path
    elif do_second_model:
        model_path = model_2_path
        model_figure_path = model_2_figure_path
        model_log_path = model_2_log_path
    elif do_third_model:
        model_path = model_3_path
        model_figure_path = model_3_figure_path
        model_log_path = model_3_log_path

    # Do model training or load model
    if do_train_model:

        # Step 5
        # Baseline Model

        if do_first_model:

            model = modelTools.model_1(
                input_shape=(256,256,13),
                num_classes=17,
                resnet_depth=100,
                resnet_filters=16
            )

        # Step 6
        # Overfit Model

        if do_second_model:
        
            # TODO : 

            pass

        # Step 7
        # Regularize Model

        if do_third_model:
            
            # TODO :
            
            pass

        # Plot the model
        keras.utils.plot_model(model, to_file=model_figure_path, show_shapes=False, show_layer_names=False, show_layer_activations=False)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        # Fit the model
        model.fit(
            x=train,
            validation_data=val,
            class_weight=class_weights,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3), keras.callbacks.TensorBoard(log_dir=model_log_path), modelTools.ClearMemory()],
            workers=8,
            use_multiprocessing=True
        )

        model.save(model_path)
    else:
        model = keras.models.load_model(model_path)
        pass

    # Predict val set
    x : SEN12MSSequence = None
    do_test : bool = False
    if do_test:
        x = test
    else:
        x = val

    y_pred = model.predict(
        x=x, 
        workers = 8, 
        use_multiprocessing=True
    )

    # Extract corresponding x, y, y_pred triplet
    x, y = x.get_item(0)
    y_pred = np.argmax(y_pred[0], axis=2)
    print(x.shape, y.shape, y_pred.shape)
    SEN12MSDataTools.plot_prediction(x, y, y_pred, os.path.join(figure_base_path, "prediction_1.png"))

if __name__ == "__main__":
    main()