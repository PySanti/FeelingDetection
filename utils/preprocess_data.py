from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import Normalizer
import numpy as np
from collections import Counter



def vectorize(seqs, num_words):
    """
        Convierte los conjuntos de resenias para que cada resenia
        sea un vector de num_words dimensiones relleno de 0s,
        pero con 1s en los indices equivalenes a los valores de las
        palabras contenidas en la resenia original

        ej:

        Resenia original : [6, 4, 3]
        Resenia convertida: [0,0,0,1,1,0,1,...]
    """
    results = np.zeros((len(seqs), num_words))
    for i, seq in enumerate(seqs):
        for a in seq:
            results[i, a] += 1.
    return results

def preprocess_data(X_train, X_test, X_val, num_words):
    """
        Recibe los conjuntos retornados por keras.datasets.imdb
        y aplica los preprocesamientos necesarios
    """
    X_train = vectorize(X_train, num_words)
    X_test = vectorize(X_test, num_words)
    X_val = vectorize(X_val, num_words)
    return X_train, X_test, X_val
