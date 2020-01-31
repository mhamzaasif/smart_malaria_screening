from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.callbacks import History, EarlyStopping, TerminateOnNaN, CSVLogger, ModelCheckpoint
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from myHelpers import plot_confusion_matrix, statsMeasure
import models
from sklearn.metrics import classification_report


base_path = os.path.join('./', 'cell_images')
infected_path = os.path.join(base_path, 'Parasitized')
uninfected_path = os.path.join(base_path, 'Uninfected')

infected_images = glob.glob(infected_path+'/*.png')
uninfected_images = glob.glob(uninfected_path+'/*.png')


# Making dataframe of cell images with label

image_files = pd.DataFrame({
    'filename': infected_images + uninfected_images,
    'label': ['infected']*len(infected_images) + ['uninfected']*len(uninfected_images)
}).sample(frac=1, random_state=42).reset_index(drop=True)


# aug_base_path = os.path.join(base_path, 'aug_data')
# train_data_path = os.path.join(aug_base_path, 'train_data')
# test_data_path = os.path.join(aug_base_path, 'test_data')
# val_data_path = os.path.join(aug_base_path, 'val_data')

target_size = (125, 125)
batch_size = 32
epochs = 100

modelFileName = []
for i in range(1, 25):
    modelFileName.append('Model'+str(i))

train_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
                                                                   # rotation_range=30,
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

test_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.
                                                                  # rotation_range=30,
                                                                  # zoom_range=0.3,
                                                                  #    width_shift_range=0.2,
                                                                  #    height_shift_range=0.2,
                                                                  #  fill_mode='nearest',
                                                                  # brightness_range=[
                                                                  #     0.1, 0.9],
                                                                  # horizontal_flip=True,
                                                                  # vertical_flip=True
                                                                  )

# train_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
#                                                             		rotation_range=20,
#                                                             		zoom_range=0.15,
#                                                             		width_shift_range=0.2,
#                                                             		height_shift_range=0.2,
#                                                             		shear_range=0.15,
#                                                             		horizontal_flip=True,
#                                                                     vertical_flip = True,
#                                                                     validation_split = 0.2,
#                                                             		fill_mode="nearest")

# test_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
#                                                             		rotation_range=20,
#                                                             		zoom_range=0.15,
#                                                             		width_shift_range=0.2,
#                                                             		height_shift_range=0.2,
#                                                             		shear_range=0.15,
#                                                             		horizontal_flip=True,
#                                                                     vertical_flip = True,
#                                                             		fill_mode="nearest")

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
print(training_data.class_indices)

# for i in range(1, len(test_frames)+1):
#    if test_frames[i][1] == 'infected':
#        test_frames[i][1] = 0
#    else:
#        test_frames[i][1] = 1

# test_frames.loc[(test_frames.label == 'infected'), 'label'] = True
# test_frames.loc[(test_frames.label == 'uninfected'), 'label'] = False

for i in range(1, 2):
    # ================================== Constructing Model ==================================
    print(i)
    if i == 1:
        model = models.getModel1()
    elif i == 2:
        model = models.getModel2()
    elif i == 3:
        model = models.getModel3()
    elif i == 4:
        model = models.getModel4()
    elif i == 5:
        model = models.getModel5()
    elif i == 6:
        model = models.getModel6()
    elif i == 7:
        model = models.getModel7()
    elif i == 8:
        model = models.getModel8()
    elif i == 9:
        model = models.getModel9()
    elif i == 10:
        model = models.getModel10()
    elif i == 11:
        model = models.getModel11()
    elif i == 12:
        model = models.getModel12()
    elif i == 13:
        model = models.getModel13()
    elif i == 14:
        model = models.getModel14()
    elif i == 15:
        model = models.getModel15()
    elif i == 16:
        model = models.getModel16()
    elif i == 17:
        model = models.getModel17()
    elif i == 18:
        model = models.getModel18()
    elif i == 19:
        model = models.getModel19()
    elif i == 20:
        model = models.getModel20()
    elif i == 21:
        model = models.getModel21()
    elif i == 22:
        model = models.getModel22()
    elif i == 23:
        model = models.getModel23()
    else:
        model = models.getModel24()
    # ============================== Training And Evaluation Section ================================

    csvLogger = CSVLogger('{0}.csv'.format(modelFileName[i-1]),
                          separator=',', append=True)
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='min')
    nanTerminator = TerminateOnNaN()
    modelCheckPoint = ModelCheckpoint('{0}.h5'.format(
        modelFileName[i-1]), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    callbacks_list = [csvLogger, nanTerminator, earlystop, modelCheckPoint]

#    history = model.fit_generator(training_data,
#                                  steps_per_epoch=training_data.samples // batch_size,
#                                  validation_data=validation_data,
#                                  validation_steps=validation_data.samples//batch_size,
#                                  epochs=epochs,
#                                  verbose=1,
#                                  callbacks=callbacks_list
#                                  )
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.savefig('./'+modelFileName[i-1]+'_acc.png')
#    plt.show()
#
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.savefig('./'+modelFileName[i-1]+'_loss.png')
#    plt.show()

    model.load_weights('{0}.h5'.format(modelFileName[i-1]))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    score = model.evaluate_generator(
        testing_data, steps=testing_data.samples//batch_size, verbose=1)

    prediction = model.predict_generator(testing_data, verbose=1)
    classes = prediction.argmax(axis=-1)
    print(classes)
    temp_prediction = prediction
    temp_prediction = temp_prediction > 0.50
    temp_frames = test_frames
    temp_frames.loc[(temp_frames.label == 'infected'), 'label'] = True
    temp_frames.loc[(temp_frames.label == 'uninfected'), 'label'] = False
    statsMeasure(test_frames['label'].astype(bool), prediction.astype(bool))

    #class_names = ['No', 'Yes']

#    plot_confusion_matrix(confusion_matrix(test_frames['label'].astype(bool), prediction),
#                          class_names,
#                          title='Confusion Matrix',
#                          modelName='{0}-Confusion.png'.format(modelFileName[i-1]))


'''
prediction=model.predict_generator(testing_data, verbose=1)

prediction[prediction > 0.5]=1
prediction[prediction <= 0.5]=0

print(confusion_matrix(testing_data.classes, prediction))
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
'''
print(history.history.keys())


# test_frames.to_csv('./test_data.csv')
