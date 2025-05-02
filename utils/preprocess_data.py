from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import RobustScaler
import numpy as np

def preprocess_data(X_train, X_test, pad_length):
    """
        Recibe los conjuntos retornados por keras.datasets.imdb
        y aplica los preprocesamientos necesarios
    """
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float64")
    X_train = pad_sequences(X_train, maxlen=pad_length, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=pad_length, padding="post", truncating="post")

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
