from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import clone_model, load_model
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utilis import data_processing, performance_evaluation
import numpy as np
import os


def apply_pruning_to_CA_WSSNet(layer):
    if isinstance(layer, Conv2D):
        if layer.name in ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, (tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=sparsity, begin_step=0, end_step=-1, frequency=100)))
    elif isinstance(layer, Dense):
        if layer.name == 'dense':
            return tfmot.sparsity.keras.prune_low_magnitude(layer, (tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=sparsity, begin_step=0, end_step=-1, frequency=100)))
    return layer


def print_model_sparsity(pruned_model):
    def _get_sparsity(weights):
        return 1.0 - np.count_nonzero(weights) / float(weights.size)

    print("\nModel Sparsity Summary ({})".format(pruned_model.name))
    print("--")
    for layer in pruned_model.layers:
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            prunable_weights = layer.layer.get_prunable_weights()
            if prunable_weights:
                print("{}: {}".format(
                    layer.name, ", ".join([
                        "({}, {})".format(weight.name,
                                          str(_get_sparsity(K.get_value(weight))))
                        for weight in prunable_weights
                    ])))


# Define a callback to execute strip_pruning and save the best model
class PruningAndSaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global max_val_accuracy
        current_val_accuracy = logs.get('val_binary_accuracy')
        if current_val_accuracy is not None:
            if current_val_accuracy >= max_val_accuracy:
                print_model_sparsity(self.model)
                strip_pruning_model = strip_pruning(self.model)
                max_val_accuracy = current_val_accuracy
                print("max_val_accuracy:", max_val_accuracy)
                # Save the best model
                strip_pruning_model.save('pruned_model.h5')
                print('Pruned model has been saved')


if __name__ == '__main__':

    # GPU usage setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    max_epoch = 400
    lr = 0.00003
    batch_size = 20
    patience = 30
    sparsity = 0.9
    Ks = 12

    # load data
    filename1 = f'data/K={Ks},dataset.pkl'
    filename2 = f'data/K={Ks},labelset.pkl'

    x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs = data_processing(filename1, filename2)

    model = load_model('model.h5')
    pruned_model = clone_model(model, clone_function=apply_pruning_to_CA_WSSNet, )
    pruned_model.summary()
    print_model_sparsity(pruned_model)

    # Initialize best validation accuracy
    max_val_accuracy = 0.0
    early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=patience)
    pruned_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['binary_accuracy'])
    pruned_model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=1, shuffle=True,
                     validation_data=(x_val, y_val),
                     callbacks=[tfmot.sparsity.keras.UpdatePruningStep(), early_stopping, PruningAndSaveCallback()])

    model = load_model('pruned_model.h5')
    model.summary()
    for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            non_zero_params = [tf.reduce_sum(tf.math.count_nonzero(w)) for w in weights]
            non_zero_params = sum(non_zero_params)
            print(f"Layer: {layer.name}, Non-zero Parameters: {non_zero_params}")
    save_path = 'pruned_result.xlsx'
    performance_evaluation(save_path, x_test, y_test, test_SNRs, model)
