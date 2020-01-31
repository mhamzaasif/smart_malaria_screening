from tensorflow import keras
import tensorflow
"""

Variant 1
Models With 3 Convolutional layers
dropout 0.5, 0.25
activations relu, tensorflow.keras.layers.LeakyReLU(alpha=0.3)

"""


def getModel1():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    # tensorflow.keras.utils.plot_model(model, to_file='model1.png',
    #                                   show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel2():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model2.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel3():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(
        125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model3.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel4():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model4.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ----------------------------------------------------------------------------------------------------------------------------
"""

Varient 2
Models With convolutions 4

"""


def getModel5():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model5.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel6():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model6.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel7():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model7.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel8():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model8.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


"""
Model with 2 Hidden layers
Same Pattern As Above

"""


"""

Variant 1
Models With 3 Convolutional layers
dropout 0.5, 0.25
activations relu, tensorflow.keras.layers.LeakyReLU(alpha=0.3)

"""


def getModel9():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model9.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel10():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model10.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel11():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model11.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel12():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model12.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ----------------------------------------------------------------------------------------------------------------------------
"""

Varient 2
Models With convolutions 4

"""


def getModel13():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model13.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel14():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model14.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel15():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model15.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel16():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model16.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


"""

Models With 3 Hidden Layers

"""

"""

Variant 1
Models With 3 Convolutional layers
dropout 0.5, 0.25
activations relu, tensorflow.keras.layers.LeakyReLU(alpha=0.3)

"""


def getModel17():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model17.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel18():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model18.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel19():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model19.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel20():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model20.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# ----------------------------------------------------------------------------------------------------------------------------
"""

Varient 2
Models With convolutions 4

"""


def getModel21():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model21.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel22():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model22.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel23():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model23.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def getModel24():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), input_shape=(125, 125, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation=tensorflow.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=128, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    tensorflow.keras.utils.plot_model(model, to_file='model24.png',
                                      show_shapes=True, show_layer_names=True)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model
