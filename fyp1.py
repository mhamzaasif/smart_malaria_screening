from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

base_path = os.path.join('/home/ucp/Documents/fyp', 'cell_images')
infected_path = os.path.join(base_path, 'Parasitized')
uninfected_path = os.path.join(base_path, 'Uninfected')

infected_images = glob.glob(infected_path+'/*.png')
uninfected_images = glob.glob(uninfected_path+'/*.png')


image_files = pd.DataFrame({
    'filename': infected_images + uninfected_images,
    'label': ['infected']*len(infected_images) + ['uninfected']*len(uninfected_images)
}).sample(frac=1, random_state=42).reset_index(drop=True)


aug_base_path = os.path.join(base_path, 'aug_data')
train_data_path = os.path.join(aug_base_path, 'train_data')
test_data_path = os.path.join(aug_base_path, 'test_data')
val_data_path = os.path.join(aug_base_path, 'val_data')

target_size = (125, 125)
batch_size = 32
train_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                                   #    rotation_range=30,
                                                                   # zoom_range=0.3,
                                                                   #    width_shift_range=0.2,
                                                                   #    height_shift_range=0.2,
                                                                   #    fill_mode='nearest',
                                                                   # horizontal_flip=True,
                                                                   # vertical_flip=True,
                                                                   # brightness_range=[
                                                                   #    0.1, 0.9],
                                                                   #    channel_shift_range=150.0,
                                                                   validation_split=0.2)
test_datagenerator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.)
                                                                  # rotation_range=30,
                                                                  # zoom_range=0.3,
                                                                  #    width_shift_range=0.2,
                                                                  #    height_shift_range=0.2,
                                                                  #  fill_mode='nearest',
                                                                  # brightness_range=[
                                                                 #     0.1, 0.9],
                                                                  # horizontal_flip=True)
train_frames, test_frames = train_test_split(
    image_files, test_size=0.2, random_state=42)

training_data = train_datagenerator.flow_from_dataframe(dataframe=train_frames, x_col='filename', y_col='label', target_size=target_size, color_mode='rgb', classes=[
                                                        'infected', 'uninfected'], class_mode='binary', batch_size=batch_size, shuffle=True, seed=42, subset='training', interpolation='nearest')

# ---------------------------------------------------- Generating validation data ------------------------------------------------------
validation_data = train_datagenerator.flow_from_dataframe(dataframe=train_frames, x_col='filename', y_col='label', target_size=target_size, color_mode='rgb', classes=[
                                                          'infected', 'uninfected'], class_mode='binary', batch_size=batch_size, shuffle=True, seed=42, subset='validation', interpolation='nearest')

# #---------------------------------------------------- Generating testing data ----------------------------------------------------------
testing_data = test_datagenerator.flow_from_dataframe(dataframe=test_frames, x_col='filename', y_col='label', target_size=target_size, color_mode='rgb', classes=[
                                                      'infected', 'uninfected'], class_mode='binary', batch_size=batch_size, shuffle=False, seed=42, interpolation='nearest')

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=(
    3, 3), input_shape=(125, 125, 3), activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPool2D((2, 2)))
model.add(keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3),activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPool2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit_generator(training_data, steps_per_epoch=training_data.samples // batch_size,
                              validation_data=validation_data, validation_steps=validation_data.samples//batch_size, epochs=10, verbose=1)
eval_history=model.evaluate_generator(
    testing_data, steps=testing_data.samples//batch_size, verbose=1)
prediction=model.predict_generator(testing_data, verbose=1)

print(prediction)
print(test_frames.label)
prediction[prediction > 0.5]=1
prediction[prediction <= 0.5]=0

print(confusion_matrix(testing_data.classes, prediction))
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
