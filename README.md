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

Un vector de longitud **variable** que representa la resenia. Cada numero representa una palabra.

En este contexto, los numeros que representan a las palabras, nacen de la frecuencia de aparicion en el dataset.

Ademas, keras te permite a traves del parametro `num_words` en el metodo `.load_data()`, controlar la riqueza del vocabulario, ya que, con `num_words` especificas la cantidad maxima de palabras unicas, dada su frecuencia de aparicion, ejemplo:

Si `num_words = 10.000`, estaran las palabras cuyo indice de aparicion sea `<= 10.000`, mayor riqueza del vocabulario, registros mas nutritivos para la red.

Si `num_words = 2`, estaran las palabras cuyo indice de aparicion sea `<= 2`, menor riqueza, registros menos nutritivos.

Cabe destacar que, el modificar el `num_words` no cambiara la logintud promedio de las resenias, simplemente, las palabras que no entren dentro del rango establecido por `num_words` seran cambiadas por un numero reservado, en este caso **2**.

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

Redividimos el conjunto, para hacer que el `X_train` tuviese 35.000 y `val + test` 15.000 (50/50).


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

Basicamente, todas las resenias con una longitud menor a `pad_length` se rellena con 0s.

Todas las resenias con una longitud mayor a `pad_length` se cortan.

El valor de `pad_length` a utilizar (en principio), sera **600**. Despues de investigar, vimos que **600 es el valor de percentil 95 para las longitudes de las resenias**.

```

# codigo

(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=15_000)
lengths = [len(i) for i in X_train]+[len(i) for i in X_test]
print(np.percentile(lengths, 95))

# salida

598.0

```

Ademas, normalizamos de una vez.


```
# codigo del main.py antes de hypertunning

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

```


## Entrenamiento

Despues de ejecutar las primeras pruebas de hypertunning utilizando `keras_tuner` y `Hyperband` como algoritmo de busqueda, atraves del siguiente codigo:

```
# main.py

from tensorflow.keras import datasets
from utils.preprocess_data import preprocess_data
from utils.split_dataset import split_dataset
import matplotlib.pyplot as plt
from keras_tuner import Hyperband 
from utils.model_builder import model_builder
import numpy as np


res_len = 600
(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data(num_words=15_000)
(X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = split_dataset(X_train, Y_train, X_test, Y_test)
X_train, X_test, X_val = preprocess_data(X_train, X_test, X_val, pad_length=res_len)



tuner = Hyperband(
    model_builder(res_len),
    factor=3,
    max_epochs=20,
    objective="val_precision",
    directory="train_results",
    project_name="FeelingsDetection"
)

tuner.search(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val)
)


# utils/model_builder.py

from tensorflow import keras
from keras import layers, metrics

def model_builder(res_len):
    def clousure(hp):
        net = keras.Sequential()

        n_hidden_layers = hp.Int('n_hidden_layers', min_value=1, max_value=5, step=1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

        net.add(layers.InputLayer(shape=(res_len,)))

        # busqueda de numero de capas optimo
        for i in range(n_hidden_layers):
            n_units = hp.Int(f'units_for_{i}', min_value=24, max_value=600, step=24)

            # busqueda de numero de neuronas optimo por capa
            net.add(layers.Dense(units=n_units, activation="relu"))

        net.add(layers.Dense(units=1, activation='sigmoid'))

        net.compile(
            loss="crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy', 'precision']
        )
        return net
    return clousure


```

Obtuvimos los siguientes resultados:

```
Trial 26 Complete [00h 02m 09s]
val_precision: 0.5098376870155334

Best val_precision So Far: 0.513620913028717
Total elapsed time: 00h 24m 18s
```

Resultados realmente lamentables.

Seguramente el problema se esta dando en la representacion de los datos: el formato utilizado para nutrir a la red con las resenias no esta siendo lo suficientemente explicativo.


