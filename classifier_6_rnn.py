##
## RNN model taking each 'slice' of incoming spectrogram image as a vector (of image's height length)
##

img_size = (50,100)
#(img_height, img_width)

epochs = 5

batch_size = 16
validation_split = 0.3
RESCALE = 1. / 255

FOLDER = 'saved_images_square_224'

classes_names = ['lp', 'tc', 'tr', 'vt']
num_classes = len(classes_names)
folders = ['data/'+FOLDER+'/train/lp/', 'data/'+FOLDER+'/train/tc/', 'data/'+FOLDER+'/train/tr/', 'data/'+FOLDER+'/train/vt/',
           'data/' + FOLDER + '/test/lp/', 'data/' + FOLDER + '/test/tc/', 'data/' + FOLDER + '/test/tr/',
           'data/' + FOLDER + '/test/vt/',
        ]
labels_texts = classes_names+classes_names
labels = [0, 1, 2, 3, 0, 1, 2, 3]

SHUFFLE_SEED=None
GRAY = True

############ Whats bellow doesn't have to be changed dramatically
# ==============================================================================
# ==============================================================================
# ==============================================================================

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
    imgs_arr = [img_to_array(load_img(path, grayscale=GRAY, target_size=target_size)) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr

for i,folder in enumerate(folders):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = sorted(onlyfiles)
    label = labels[i]
    paths = [folder+file for file in onlyfiles]
    Y_all_labels += [label]*len(paths)
    X_all_paths += paths

X_all_image_data = load_images_with_keras(X_all_paths, target_size=img_size)
Y_all_labels = np.array(Y_all_labels)

Y_all_labels = keras.utils.to_categorical(Y_all_labels, num_classes=num_classes)

print("X_all_image_data:", X_all_image_data.shape)
print("Y_all_labels:", Y_all_labels.shape)
print("---")

def shuffle_two_lists_together(a,b, SEED=None):
    if SEED is not None:
        random.seed(SEED)

    sort_order = list(range(0,len(a)))
    random.shuffle(sort_order)

    a_new = [a[i] for i in sort_order]
    b_new = [b[i] for i in sort_order]
    a_new = np.asarray(a_new)
    b_new = np.asarray(b_new)
    return a_new, b_new

def split_data(x,y,validation_split=0.2):
    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]

    print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,x_test,y_test

X_all_image_data,Y_all_labels = shuffle_two_lists_together(X_all_image_data,Y_all_labels,SEED=SHUFFLE_SEED)
x_train,y_train,x_test,y_test = split_data(X_all_image_data,Y_all_labels,validation_split=validation_split)

# WE NEED TO GO FAST SONIC
#x_train = x_train[0:20]
#y_train = y_train[0:20]
#x_test = x_test[0:20]
#y_test = y_test[0:20]

x_train *= RESCALE
x_test *= RESCALE

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)#, y_train[0:10])
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

# =============================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# SHAPE is: samples_2063 , time_img_width_150 , features_img_height
#x_test: (2063, 150, 200, 1)


x_train = x_train.reshape(x_train.shape[:-1])
x_test = x_test.reshape(x_test.shape[:-1])
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)

# MODEL
from keras import optimizers
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
##model.add(LSTM(500, input_shape=x_train.shape[1:], return_sequences=True))
##model.add(LSTM(500))
model.add(LSTM(150, input_shape=x_train.shape[1:]))
model.add(Dense(128))
#model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(num_classes))

# ARCHITECTURE changeable....

model.summary()

# maybe too agressive? eventually gets to arround 0.45 accuracy, which is not the best...
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# well also not so hot...
# we seem to be missing the extra info extracted by already pretrained model (in the initial epochs at least)
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

# ==============================================================================
# TRAIN 1
# ==============================================================================
#

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

from visualize_history import visualize_history
specialname = ''
visualize_history(history.history, show_also='acc', save=True, save_path='classifier6_'+str(epochs)+'epochs_'+specialname)
