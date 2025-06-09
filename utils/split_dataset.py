from sklearn.model_selection import train_test_split
import numpy as np


def split_dataset(X_train, Y_train, X_test, Y_test):
    """
        25.000 registros para test-validacion es demasiado,
        esta funcion junta los datos retornados por .load_data
        los une y luego los divide a traves de train_test_split
    """

    X_data = np.concatenate((X_train, X_test), axis=0)
    Y_data = np.concatenate((Y_train, Y_test), axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=42, stratify=Y_data, test_size=.3)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, random_state=42, stratify=Y_test, test_size=.5)

    return ((X_train, Y_train), (X_test, Y_test), (X_val, Y_val))





