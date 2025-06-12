from os import wait
from keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import datasets
from utils.preprocess_data import preprocess_data
from utils.show_train_results import show_train_results
from utils.split_dataset import split_dataset
import matplotlib.pyplot as plt
from keras_tuner import Hyperband 
from utils.model_builder import model_builder
import numpy as np


num_words = 10_000
(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=num_words)
(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = split_dataset(X_train, Y_train, X_test, Y_test)
X_train, X_test, X_val = preprocess_data(X_train, X_test, X_val, num_words=num_words)



tuner = Hyperband(
    model_builder(num_words),
    factor=2,
    max_epochs=15,
    objective="val_precision",
    directory="train_results",
    project_name="FeelingsDetection"
)

tuner.search(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val)
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Mejor configuracion de hiperparametros ")
print(best_hps.values)

early_stopping = EarlyStopping(
    monitor='val_precision',   # Métrica a monitorear (puede ser 'val_accuracy')
    patience=15,          # Número de épocas sin mejora antes de detener
    restore_best_weights=True , # Restaura los pesos del mejor modelo
    mode="max",
    verbose=1
)

model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,  # Puedes ajustar según los resultados del tuner
    callbacks=[early_stopping]
)

test_loss, test_accuracy, test_precision = model.evaluate(X_test, Y_test)
print(f"""
Resultados para conjunto de test

Loss : {test_loss}
Accuracy : {test_accuracy}
Precision : {test_precision}
""")
show_train_results(history)
