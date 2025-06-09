from tensorflow.keras import datasets
from utils.preprocess_data import preprocess_data
from utils.split_dataset import split_dataset
import matplotlib.pyplot as plt
from keras_tuner import Hyperband 
from utils.model_builder import model_builder
import numpy as np


(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=15_000)
(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = split_dataset(X_train, Y_train, X_test, Y_test)
X_train, X_test, X_val = preprocess_data(X_train, X_test, X_val, pad_length=600)



tuner = Hyperband(
    model_builder,
    factor=3,
    max_epochs=20,
    objective="val_precision"
)
