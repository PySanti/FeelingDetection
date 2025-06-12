from tensorflow import keras
from keras import layers, metrics
from keras import regularizers

def model_builder(num_words):
    def clousure(hp):
        net = keras.Sequential()

        n_hidden_layers = hp.Int('n_hidden_layers', min_value=1, max_value=2, step=1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        net.add(layers.InputLayer(shape=(num_words,)))

        # busqueda de numero de capas optimo
        for i in range(n_hidden_layers):
            n_units = hp.Int(f'units_for_{i}', min_value=48, max_value=576, step=24)
            regu_const_value = hp.Choice(f'regu_const_{i}', values=[1e-1, 1e-2, 1e-3])
            dropout_rate = hp.Float(f'drop_rate_{i}', min_value=0.1, max_value=0.4, step=0.05)

            # busqueda de numero de neuronas optimo por capa
            net.add(layers.Dense(
                units=n_units, 
                activation="relu", 
                kernel_regularizer=regularizers.l2(regu_const_value)))
            net.add(layers.Dropout(rate=dropout_rate))

        net.add(layers.Dense(units=1, activation='sigmoid'))

        net.compile(
            loss="crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy', 'precision']
        )
        return net
    return clousure

