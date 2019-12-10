##
##  Reference: https://github.com/anujshah1003/Transfer-Learning-in-keras---custom-data
##

import numpy as np
import json

import os

from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_last', 'channels_first'}

    if dim_ordering == 'channels_first':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def decode_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
    return results


def create_test_train_data(data_dir_list, data_path, labelmap):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    SEED = 2

    for dataset in data_dir_list:
        if dataset.startswith('.DS'):
            continue

        img_data_list = []
        img_list = os.listdir(data_path + '/' + dataset)
        labels = np.zeros(len(img_list))


        print('Loaded the images of dataset-' + '{}\n'.format(dataset))
        idx = 0
        for img in img_list:
            if img.startswith('.DS'):
                continue
            img_path = data_path + '/' + dataset + '/' + img
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            img_data_list.append(x)
            labels[idx] = labelmap[dataset]
            idx += 1

        #--- Split to get the test and train for this folder
        img_data = np.array(img_data_list)
        print(img_data.shape)
        img_data = np.rollaxis(img_data, 1, 0)
        print(img_data.shape)
        img_data = img_data[0]
        print(img_data.shape)

        # convert class labels to on-hot encoding
        num_classes = len(list(set(labelmap.values())))
        Y = np_utils.to_categorical(labels, num_classes)

        # Shuffle the dataset
        x, y = shuffle(img_data, Y, random_state=2)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

        train_data.append(X_train)
        test_data.append(X_test)
        train_label.append(y_train)
        test_label.append(y_test)

        # pickle the mean value by category
        Xtrain_mean = np.mean(X_train, axis=0)
        np.save('Avg/{}_mean'.format(dataset), Xtrain_mean)

    #--- train data
    trainX_final = np.vstack(train_data)
    print(trainX_final.shape)
    trainY_final = np.vstack(train_label)
    print(trainY_final.shape)

    #--- test data
    testX_final = np.vstack(test_data)
    print(testX_final.shape)
    testY_final = np.vstack(test_label)
    print(testY_final.shape)

    # Shuffle Training Set
    trainX_final_s, trainY_final_s = shuffle(trainX_final, trainY_final, random_state=SEED)

    #---pickle the average of all training data
    trainX_final_mean = np.mean(trainX_final_s, axis=0)
    np.save('Avg/trainX_final_mean', trainX_final_mean)

    return trainX_final_s, trainY_final_s, testX_final, testY_final

def plot_graph(hist, epoch, graphname):
    # visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    xc = range(epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.savefig('{}_loss.png'.format(graphname))

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.savefig('{}_accuracy.png'.format(graphname))
    #plt.show()




