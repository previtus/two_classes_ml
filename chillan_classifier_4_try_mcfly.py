# SETUP

img_size = None #(20,20)
# btw img_size = (28,28) would be without rounding
img_size = (22,22)
epochs = 25
batch_size = 16
validation_split = 0.3
RESCALE = 1. / 255 # put data from 0-255 into 0-1
# GET ALL DATA
# define the classes in here directly

FOLDER = 'chillan_saved_images_square_224_onlyLonger1000'
specialname = '__MODEL2_1000longer_tryingtogetbestres'
#FOLDER = 'saved_images_square_20'

classes_names = ['lp', 'tr', 'vt']
num_classes = len(classes_names)
folders = ['data/'+FOLDER+'/LP/', 'data/'+FOLDER+'/TR/', 'data/'+FOLDER+'/VT/']
labels_texts = classes_names
labels = [0, 1, 2] # list(range(0,len(classes_names)))

SHUFFLE_SEED=43
SUBSET_N = 15000
BALANCE_DATA_TO_AMOUNT = 5000 # -1 stops it
DROP=0.2
############ Whats bellow doesn't have to be changed dramatically

X_all_paths = []
Y_all_labels = []

from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import keras
from matplotlib import pyplot as plt

def load_images_with_keras(img_paths, target_size=None):
    imgs_arr = [img_to_array(load_img(path, target_size=target_size)) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr

def sample_random_subset_from_list(L, N):
    if len(L) < N:
        print("less than N=",N,"data, selecting it all (without shuffle)")
        return L
    # warn this works inplace!
    random.shuffle(L)
    S = L[0:N]
    #print("subset", S)
    return S

def shuffle_two_lists_together(a,b, SEED=None):
    if SEED is not None:
        random.seed(SEED)

    sort_order = random.sample(range(len(a)), len(a))
    #random.shuffle(range(0,len(a)))
    a_new = [a[i] for i in sort_order]
    b_new = [b[i] for i in sort_order]
    a_new = np.asarray(a_new)
    b_new = np.asarray(b_new)
    return a_new, b_new

#def shuffle_two_lists_together(a,b, SEED=None):
#    combined = list(zip(a, b))
#    if SEED is not None:
#        random.seed(SEED)
#    random.shuffle(combined)
#    a[:], b[:] = zip(*combined)
#    return a,b

def how_many_are_in_each_category(Y):
    unique_categories = set(Y)
    data_by_category = {}
    for cat in unique_categories:
        data_by_category[cat] = []
        for j in range(0,len(Y)):
            if Y[j] == cat:
                data_by_category[cat].append(Y[j])
    for cat in unique_categories:
        print(cat, " occured ", len(data_by_category[cat]))


def sample_same_amount_of_data_from_each_category(X,Y,number_to_sample):
    unique_categories = set(Y)
    print("unique_categories",unique_categories)

    data_by_category = {}
    for cat in unique_categories:
        data_by_category[cat] = []
        for j in range(0,len(Y)):
            if Y[j] == cat:
                data_by_category[cat].append(X[j])
    new_X = []
    new_Y = []
    for cat in unique_categories:
        balanced_subset = sample_random_subset_from_list(data_by_category[cat], number_to_sample)

        print(cat, " occured ", len(data_by_category[cat]), " => now its", len(balanced_subset))
        #print("balanced_subset for",cat)
        #print(balanced_subset)
        labels_subset = [cat]*len(balanced_subset)
        new_X += balanced_subset
        new_Y += labels_subset

    return new_X,new_Y # needs shuffling after!

def convert_back_from_categorical_data(Y):
    # manually written, rewrite
    # turn list of values according to this
    # (1,0,0) => 0
    # (0,1,0) => 1
    # (0,0,1) => 2
    new_Y = []
    #print("   ",Y[0], "should be", np.argmax(Y[0]))
    for y in Y:
        k = np.argmax(y)
        new_Y.append(k)
    return new_Y

for i,folder in enumerate(folders):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    print(len(onlyfiles), "loading", labels_texts[i], labels[i])
    onlyfiles = sorted(onlyfiles)
    label = labels[i]
    #print(folder+onlyfiles[0])
    #print(folder+onlyfiles[-1])
    paths = [folder+file for file in onlyfiles]
    Y_all_labels += [label]*len(paths)
    X_all_paths += paths


if BALANCE_DATA_TO_AMOUNT > -1:
    X_all_paths, Y_all_labels = sample_same_amount_of_data_from_each_category(X_all_paths, Y_all_labels, BALANCE_DATA_TO_AMOUNT)

if len(X_all_paths) > SUBSET_N:
    print("Selecting a small subset of", SUBSET_N, "from",len(X_all_paths))
    X_all_paths, Y_all_labels = shuffle_two_lists_together(X_all_paths, Y_all_labels)
    X_all_paths = X_all_paths[0:SUBSET_N]
    Y_all_labels = Y_all_labels[0:SUBSET_N]
else:
    print("We have",len(X_all_paths),"data, which is less than the subset value of",SUBSET_N)

print("Loading image data!")
X_all_image_data = load_images_with_keras(X_all_paths, target_size=img_size)
Y_all_labels = np.array(Y_all_labels)
Y_all_labels = keras.utils.to_categorical(Y_all_labels, num_classes=num_classes)

print("X_all_image_data:", X_all_image_data.shape)
print("Y_all_labels:", Y_all_labels.shape)
print("---")

VIZ = False
if VIZ:
    images = range(0,9)
    for i in images:
        plt.subplot(330 + 1 + i)
        plt.imshow(X_all_image_data[i])
    #Show the plot
    plt.show()


VIZ = False
if VIZ:
    plt.hist(Y_all_labels, alpha=0.5)
    plt.title('Number of examples from each class')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')
    plt.show()

# NOW WE HAVE ALL THE DATA X AND THEIR LABELS Y IN X_all_image_data, Y_all_labels
def split_data(x,y,validation_split=0.2):
    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]

    print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,x_test,y_test

print("SanityCheck whole dist:")
how_many_are_in_each_category(convert_back_from_categorical_data(Y_all_labels))

for i in range(0,5):
    print(Y_all_labels[i])

print("Shuffling all data now...")
Y_all_labels = np.asarray(Y_all_labels)
X_all_image_data,Y_all_labels = shuffle_two_lists_together(X_all_image_data,Y_all_labels,SEED=SHUFFLE_SEED)

for i in range(0,5):
    print(Y_all_labels[i])

# Is the shuffling bonked???
how_many_are_in_each_category(convert_back_from_categorical_data(Y_all_labels))

x_train,y_train,x_test,y_test = split_data(X_all_image_data,Y_all_labels,validation_split=validation_split)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)#, y_train[0:10])
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

print("SanityCheck Test dist:")
how_many_are_in_each_category(convert_back_from_categorical_data(y_test))
print("SanityCheck Train dist:")
how_many_are_in_each_category(convert_back_from_categorical_data(y_train))
print("---")

VIZ = False
if VIZ:
    images = range(0,9)
    for i in images:
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i])
    #Show the plot
    plt.show()

# See how is the distribution in test and train (should be the same)
VIZ = False
if VIZ:
    from matplotlib import pyplot as plt
    plt.subplot(2, 1, 1)
    plt.hist(y_train, alpha=0.5)
    plt.title('Train data classes')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')

    plt.subplot(2, 1, 2)
    plt.hist(y_test, alpha=0.5)
    plt.title('Test data classes')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')
    plt.show()

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# Now for the model


import sys
import os
import numpy as np
import pandas as pd
# mcfly
from mcfly import modelgen, find_architecture, storage
from keras.models import load_model
np.random.seed(2)
from keras.utils import plot_model

# mc fly has
# X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary, labels
## X_train shape: (11397, 512, 9)
## y_train_binary shape: (11397, 7)

# we have
## x_train: (8181, 150, 150, 3)
## y_train: (8181, 3)
## x_test: (3507, 150, 150, 3)
## y_test: (3507, 3)

X_train = x_train
y_train_binary = y_train
X_val = x_test
y_val_binary = y_test


## Little bit hacky flattening!

num_classes = y_train_binary.shape[1]
print("num_classes:", num_classes)

X_train = X_train.flatten().reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2], X_train.shape[3])
X_val = X_val.flatten().reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2], X_val.shape[3])
print(X_train.shape)
print(X_val.shape)

#input_shape = (num_samples, num_timesteps, num_channels)
models = modelgen.generate_models(X_train.shape,
                                  number_of_classes=num_classes,
                                  number_of_models = 20)
directory_to_extract_to = '.'
resultpath = os.path.join(directory_to_extract_to, 'data/models')
modelimgpath = os.path.join(directory_to_extract_to, 'data/models/img_models')
if not os.path.exists(resultpath):
        os.makedirs(resultpath)
if not os.path.exists(modelimgpath):
        os.makedirs(modelimgpath)

VIZ = True
if VIZ:
    models_to_print = range(len(models))
    for i, item in enumerate(models):
        if i in models_to_print:
            model, params, model_types = item
            print(
                "-------------------------------------------------------------------------------------------------------")
            print("Model " + str(i))
            print(" ")
            print("Hyperparameters:")
            print(params)
            print(" ")
            print("Model description:")
            model.summary()
            plot_model(model, to_file=modelimgpath+'/model_'+str(i)+'_'+model_types+'.png', show_shapes=True)

            print(" ")
            print("Model type:")
            print(model_types)
            print(" ")




# TRAIN!
outputfile = os.path.join(resultpath, 'modelcomparison.json')
histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(X_train, y_train_binary,
                                                                           X_val, y_val_binary,
                                                                           models,nr_epochs=10,
                                                                           subset_size=2000,
                                                                           verbose=True,
                                                                           outputfile=outputfile)
print('Details of the training process were stored in ',outputfile)

modelcomparisons = pd.DataFrame({'model':[str(params) for model, params, model_types in models],
                       'train_acc': [history.history['acc'][-1] for history in histories],
                       'train_loss': [history.history['loss'][-1] for history in histories],
                       'val_acc': [history.history['val_acc'][-1] for history in histories],
                       'val_loss': [history.history['val_loss'][-1] for history in histories]
                       })
modelcomparisons.to_csv(os.path.join(resultpath, 'modelcomparisons.csv'))

print(modelcomparisons)