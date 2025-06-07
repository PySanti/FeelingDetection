# FeelingDetection

El objetivo de este proyecto será crear una red neuronal capaz de identificar si una reseña de una película en IMDB es positiva (1) o negativa (0).

Para ello utilizaremos el conjunto de datos provisto por la librería `Keras`, específicamente `keras.datasets.imdb`.

La idea de este proyecto surge de la necesidad de practicar y jugar un poco con redes neuronales.

Se utilizará una **fully connected network** tipo **MLP** para este ejercicio. La cantidad de capas y neuronas se irá decidiendo a medida que vayamos probando.

Luego, se utilizará como función de activación `Relu` en las *hidden layers* y `Sigmoid` en la neurona del *output layer* al ser un problema de clasificación.


## Preprocesamiento

Shape del dataset completo : `50000`

Es destacable mencionar que, el dataset contiene las resenias ya vectorizadas. La forma que utiliza keras para vectorizar las resenias es basicamente cambiar cada palabra por un numero que representa la inversa de su frecuencia de aparicion promedio en algo llamado corpus, que es basicamente un conjunto de registros de texto muy grande.

El conjunto cuenta con `50000` resenias, sin embargo, las resenias a pesar de estar ya vectorizadas, no tienen todas las misma longitud y ademas estan contenidas dentro de objetos tipo lista de python, lo que genera que el X_train y X_test sean vectores de una sola dimension donde cada entrada es una lista de python.

Para solucionar los dos problemas anteriores utilizamos el siguiente codigo

```
from tensorflow.keras import datasets
from keras.preprocessing.sequence import pad_sequences

max_length = 500
(X_train, Y_train), (X_test, Y_test) = datasets.imdb.load_data()
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")
```

La funcion pad_sequences rellena los espacios vacios de las resenias con un 0, de tal modo todas los vectores que representan a las resenias tengan la misma longitud.

Es destacable mencionar que, a partir de ahora, cada resenia va a estar representada por un vector de 500 elementos, dicho valor fue seleccionado de manera completamente arbitraria.

### Escalado

Se opto por normalizacion en el escalado.

## Entrenamiento y evaluación

Ademas de utilizar pad_sequences, tambien utilizaremos embeddings: una tecnica que transforma cada numero (palabra) de la resenia en un vector. De este modo, las palabras semanticamente parecidas (alegre, feliz, contento, etc) estaran representadas por vectores similares. Esta tecnica permite que la red encuentre relaciones mucho mas  complejas entre las palabras.

Este es un tipo de problema que requiere muchos mas conocimientos de los que tenemos en este momento, asi que decidi dejarlo para mas adelante.

Volvere.
