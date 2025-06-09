# FeelingDetection

El objetivo de este proyecto será crear una red neuronal capaz de identificar si una reseña de una película en IMDB es positiva (1) o negativa (0).

Para ello utilizaremos el conjunto de datos provisto por la librería `Keras`, específicamente `keras.datasets.imdb`.

La idea de este proyecto surge de la necesidad de practicar y jugar un poco con redes neuronales.

Se utilizará una **fully connected network** tipo **MLP** para este ejercicio. La cantidad de capas y neuronas se irá decidiendo a medida que vayamos probando.

Luego, se utilizará como función de activación `Relu` en las *hidden layers* y `Sigmoid` en la neurona del *output layer* al ser un problema de clasificación.


## Preprocesamiento

Longitud del dataset completo : `50000`

El dataset esta compuesto por resenias y un target que representa si la resenia es positiva (1) o negativa (0).


Cada resenia tiene el siguiente formato:

```
[1, 5, 198, 138, 254, 8, 967, 10, 10, 39, 4, 1158, 213, 7, 650, 7660, 1475, 213, 7, 650, 13, 215, 135, 13, 1583, 754, 2359, 133, 252, 50, 9, 49, 1104, 136, 32, 4, 1109, 304, 133, 1812, 21, 15, 191, 607, 4, 910, 552, 7, 229, 5, 226, 20, 198, 138, 10, 10, 241, 46, 7, 158]
```

Un vector de longitud **variable** que representa la resenia. Cada numero representa una palabra

En este contexto, los numeros que representan a las palabras, nacen de la frecuencia de aparicion en el dataset.

Ademas, keras te permite a traves del parametro `num_words` en el metodo `.load_data()`, controlar la riqueza del vocabulario, ya que, con `num_words` especificas la cantidad maxima de palabras unicas, dada su frecuencia de aparicion, ejemplo:

Si num_words = 10.000, estaran las palabras cuyo indice de aparicion sea <= a 10.000, mayor riqueza del vocabulario, registros mas nutritivos para la red.

Si num_words = 2, estaran las palabras cuyo indice de aparicion sea <= 2, menor riqueza, registros menos nutritivos.

Cabe destacar que, el modificar el num_words no cambiara la logintud promedio de las resenias, simplemente, las palabras que no entren dentro del rango establecido por `num_words` seran cambiadas por un numero reservado, en este caso **2**.

Empezaremos utilizando `num_words = 15.000`.

Luego, a traves del siguiente codigo:

```
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

```

Redividimos el conjunto, para hacer que el X_train tuviese 35.000 y val + test 15.000.


Luego, tuvimos que hacer que todas las resenias tuviesen la misma longitud, esto lo hicimos a traves de la funcion `pad_sequences`:

```
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import Normalizer
import numpy as np

def preprocess_data(X_train, X_test, X_val, pad_length):
    """
        Recibe los conjuntos retornados por keras.datasets.imdb
        y aplica los preprocesamientos necesarios
    """
    X_train = pad_sequences(X_train, maxlen=pad_length, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=pad_length, padding="post", truncating="post")
    X_val = pad_sequences(X_val, maxlen=pad_length, padding="post", truncating="post")
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    X_val = np.array(X_val).astype("float32")

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_test)
    return X_train, X_test, X_val

```

Ademas, normalizamos de una vez.

El valor de `pad_length` a utilizar (en principio), sera 300. Despues de investigar, vimos que ~95% de las resenias tienen una longitud <= 500.



## Entrenamiento