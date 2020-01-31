#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 00:29:41 2018

@author: awais
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import top_k_categorical_accuracy
import os


def plot_history(history, modelName=''):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    # plt.figure()
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    N = len(history.history["loss"])
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plotFile = 'plot.png'

    if modelName != '':
        plotFile = '{0}_plot.png'.format(modelName)

    plt.savefig(fname=plotFile, dpi=300)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          modelName='',
                          large=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, Without Normalization')

    print(cm)

    font_prop = font_manager.FontProperties(size=20)
    font_prop1 = font_manager.FontProperties(size=16)

    plt.figure(figsize=(8, 6), dpi=80)
    if large == True:
        plt.figure(figsize=(12, 9), dpi=80)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontproperties=font_prop)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes, rotation='vertical',
               horizontalalignment="center", verticalalignment="center")
    plt.tick_params(axis='x', grid_alpha=0)
    plt.tick_params(axis='y', grid_alpha=0)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 verticalalignment="center",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontproperties=font_prop1)

    plt.tight_layout()
    plt.ylabel('Actual Label', fontproperties=font_prop1)
    plt.xlabel('Predicted Label', fontproperties=font_prop1)

    plotFile = 'matrix.png'
    if modelName != '':
        plotFile = '{0}_matrix.png'.format(modelName)
    plt.savefig(fname=plotFile, dpi=300, bbox_inches="tight")
    plt.close()


def statsMeasure(values, preds):
    cm = confusion_matrix(values, preds)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]

    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spc = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print('accuracy\tsensitivity\tspecificity\tppv\tnpv')
    print('{0} %\t{1} %\t{2} %\t{3} %\t{4} %'.format(
        acc*100, sen*100, spc*100, ppv*100, npv*100))
#    print('Accuracy = {0} %'.format(acc*100))
#    print('Sensitivity = {0} %'.format(sen*100))
#    print('Specificity = {0} %'.format(spc*100))


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
