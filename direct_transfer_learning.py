from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from utilis import tl_data_processing, performance_evaluation
import os

if __name__ == '__main__':
    # GPU usage setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    max_epoch = 50
    batch_size = 10
    patience = 10
    lr = 0.00003
    data_num = 100
    Kt = 8

    # load data
    filename1 = f'data/K={Kt},dataset.pkl'
    filename2 = f'data/K={Kt},labelset.pkl'

    x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs = tl_data_processing(filename1, filename2,
                                                                                             data_num)

    model = load_model('model.h5')
    for layer in model.layers:
        if layer.name == 'dense' or layer.name == 'dense_1':
            layer.trainable = True
            continue
        layer.trainable = False
    early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=patience)
    best_model_path = 'direct_transfer_learning_model.h5'
    checkpointer = ModelCheckpoint(best_model_path, verbose=1, monitor='val_binary_accuracy', save_best_only=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['binary_accuracy'])
    model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=1, shuffle=True,
              validation_data=(x_val, y_val),
              callbacks=[early_stopping, checkpointer])

    model = load_model(best_model_path)
    save_path = 'direct_transfer_learning_result.xlsx'
    performance_evaluation(save_path, x_test, y_test, test_SNRs, model)
