from tensorflow import keras
from keras import layers, metrics

def model_builder(hp, res_len):
    net = keras.Sequential()

    n_hidden_layers = hp.Int('n_hidden_layers', min_value=1, max_value=5, step=1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    net.add(layers.InputLayer(shape=(500,)))

    # busqueda de numero de capas optimo
    for i in range(n_hidden_layers):
        n_units = hp.Int(f'units_for_{i}', min_value=24, max_value=600, step=24)

        # busqueda de numero de neuronas optimo por capa
        net.add(layers.Dense(units=n_units, activation="relu"))

    net.add(layers.Dense(units=1, activation='sigmoid'))

    net.compile(
        loss="cross_entropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy', 'precision']
    )
    return net

