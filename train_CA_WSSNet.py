import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utilis import data_processing, performance_evaluation
from CA_WSSNet import CA_WSSNet
import os

# GPU usage setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

max_epoch = 500
lr = 0.00003
batch_size = 20
patience = 30
N = 64
L = 40
K = 12


# load data
filename1 = f'data/K={K},dataset.pkl'
filename2 = f'data/K={K},labelset.pkl'

x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs = data_processing(filename1, filename2)

# callbacks
early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=patience)
best_model_path = 'model.h5'
checkpointer = ModelCheckpoint(best_model_path, verbose=1, monitor='val_binary_accuracy', save_best_only=True)
model = CA_WSSNet((L, N, 2))
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['binary_accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=max_epoch, batch_size=batch_size, verbose=1, shuffle=True,
          validation_data=(x_val, y_val), callbacks=[early_stopping, checkpointer])

model = load_model(best_model_path)
save_path = 'result.xlsx'
performance_evaluation(save_path, x_test, y_test, test_SNRs, model)