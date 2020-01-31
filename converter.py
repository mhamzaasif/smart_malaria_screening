#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:03:28 2020

@author: ucp
"""
import os
import tensorflow
from tensorflow import keras
import pandas
from sklearn.metrics import classification_report, confusion_matrix
# import h5
import models
from contextlib import redirect_stdout
from myHelpers import plot_confusion_matrix
modelFileName = []
for i in range(1, 25):
    modelFileName.append('Model'+str(i))

base_path = os.path.join('./', "converted_models")

print(base_path)
# target_size = (125, 125)
# batch_size = 32
# test_frames = pandas.DataFrame.from_csv('./test_data.csv')
# print(test_frames)
# test_frames.loc[(test_frames.label == True), 'label'] = 'uninfect ed'
# test_frames.loc[(test_frames.label == False), 'label'] = 'infected'
# test_datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.
#                                                                   # rotation_range=30,
#                                                                   # zoom_range=0.3,
#                                                                   #    width_shift_range=0.2,
#                                                                   #    height_shift_range=0.2,
#                                                                   #  fill_mode='nearest',
#                                                                   # brightness_range=[
#                                                                   #     0.1, 0.9],
#                                                                   # horizontal_flip=True,
#                                                                   # vertical_flip=True
#                                                                   )
# testing_data = test_datagenerator.flow_from_dataframe(dataframe=test_frames, x_col='filename', y_col='label', target_size=target_size, color_mode='rgb', classes=[
#                                                       'infected', 'uninfected'], class_mode='binary', batch_size=batch_size, shuffle=False, seed=42, interpolation='nearest')
# test_frames.loc[(test_frames.label == 'uninfected'), 'label'] = True
# test_frames.loc[(test_frames.label == 'infected'), 'label'] = False
for i in range(1, 25):
    # ================================== Constructing Model ==================================

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

    # with open(modelFileName[i-1]+'.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()

    print("Converting Model: " + modelFileName[i-1])
    # input()
    # model.load_weights('{0}.h5'.format(modelFileName[i-1]))
    # model.compile(optimizer=keras.optimizers.Adam(),
    #               loss='binary_crossentropy', metrics=['accuracy'])

    # model.save(base_path+'/'+modelFileName[i-1]+'.h5', include_optimizer=True)
    filename = base_path+'/'+modelFileName[i-1] + '.h5'
    print(filename)

    converter = tensorflow.lite.TFLiteConverter.from_keras_model_file(filename)
    converter.experimental_new_converter = True
    tfmodel = converter.convert()

    open(base_path+'/'+modelFileName[i-1]+".tflite", "wb").write(tfmodel)

    # score = model.evaluate_generator(
    #     testing_data, steps=testing_data.samples//batch_size, verbose=1)

    # prediction = model.predict_generator(testing_data, verbose=1)
    # prediction = prediction > 0.50
    # print(prediction)
    # # class_report = pandas.DataFrame(classification_report(
    # #     test_frames['label'].astype(bool), prediction, output_dict=True)).transpose()
    # # class_report.to_csv('./'+modelFileName[i-1]+'report.csv')
    # class_names = ['infected', 'uninfected']
    # plot_confusion_matrix(confusion_matrix(test_frames['label'].astype(bool), prediction),
    #                       class_names,
    #                       title='Confusion Matrix',
    #                       modelName='{0}-Confusion.png'.format(modelFileName[i-1]))
