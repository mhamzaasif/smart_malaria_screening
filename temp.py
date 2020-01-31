from tensorflow import keras as keras


def get_conv(blocks=1, activation='relu', layers=1):
    if layers == 1:
        return keras.layers.Conv2d(filters=64, kernel=(3, 3), activation=activation)
        elif layers == 2:
            layer = keras.layers.Conv2D(
                filters=64, kernel=(3, 3), activation=activation)
            layer = keras.layers.Conv2D(filters=32, kernel=(
                3, 3), activation=activation)(layer)
        layer = keras.layers.Conv2D(
            filters=64, kernel=(3, 3), activation=activation)
        layer = keras.layers.Conv2D(filters=32, kernel=(
            3, 3), activation=activation)(layer)


def get_dense(layers=1)
