# SETUP

img_size = None #(20,20)
img_size = (150,150)
epochs = 30
batch_size = 16

validation_split = 0.3

RESCALE = 1. / 255 # put data from 0-255 into 0-1
#RESCALE = 0.004
#RESCALE = 0.01
#RESCALE = 0

# GET ALL DATA
# define the classes in here directly

FOLDER = 'saved_images_square_224'
#FOLDER = 'saved_images_square_20'

classes_names = ['lp', 'tc', 'tr', 'vt']
num_classes = len(classes_names)
folders = ['data/'+FOLDER+'/train/lp/', 'data/'+FOLDER+'/train/tc/', 'data/'+FOLDER+'/train/tr/', 'data/'+FOLDER+'/train/vt/',
           'data/' + FOLDER + '/test/lp/', 'data/' + FOLDER + '/test/tc/', 'data/' + FOLDER + '/test/tr/',
           'data/' + FOLDER + '/test/vt/',
        ]
labels_texts = classes_names+classes_names
labels = [0, 1, 2, 3, 0, 1, 2, 3]

SHUFFLE_SEED=13


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

for i,folder in enumerate(folders):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = sorted(onlyfiles)
    label = labels[i]
    #print(len(onlyfiles), "loaded", labels_texts[i], labels[i])
    #print(folder+onlyfiles[0])
    #print(folder+onlyfiles[-1])
    paths = [folder+file for file in onlyfiles]
    Y_all_labels += [label]*len(paths)
    X_all_paths += paths

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
def shuffle_two_lists_together(a,b, SEED=None):
    combined = list(zip(a, b))
    if SEED is not None:
        random.seed(SEED)
    random.shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a,b

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
print("x_test:", x_train.shape)
print("y_test:", y_train.shape)#, y_train[0:10])
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
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

# Now for the model

"""
# LOAD AND TRAIN FULL MODEL
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input

# LOAD VGG16
input_tensor = Input(shape=(img_size[0],img_size[1],3))
model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_tensor=input_tensor)


# CREATE A TOP MODEL
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='sigmoid'))


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)

# CONCATENATE THE TWO MODELS
new_model.add(top_model)

# LOCK THE TOP CONV LAYERS
for layer in new_model.layers[:15]:
#for layer in new_model.layers:
    layer.trainable = False

# COMPILE THE MODEL
new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', #optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
new_model.summary()
model = new_model

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE)

# Show what we want to train on?

VIZ=False
if VIZ:
    img_rows, img_cols = img_size
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        print(np.asarray(x_batch[0]).shape)
        print(x_batch[0].shape)

        # Show the first 9 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(x_batch[i].reshape(img_rows, img_cols, 3))
        # show the plot
        plt.show()
        break
if VIZ:
    img_rows, img_cols = img_size
    for x_batch, y_batch in datagen.flow(x_test, y_test, batch_size=9):
        print(np.asarray(x_batch[0]).shape)
        print(x_batch[0].shape)

        # Show the first 9 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(x_batch[i].reshape(img_rows, img_cols, 3))
        # show the plot
        plt.show()
        break

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=datagen.flow(x_test, y_test, batch_size=batch_size),
                        #validation_data=(x_test, y_test),
                        verbose=1)

"""


# WITH SAVED FEATURES - much faster
# but I had some doubts about the augmentation on/off on train and test data

from keras import applications
model = applications.VGG16(include_top=False, weights='imagenet')

# HERE WE ACTUALLY HAVE TO EDIT THE DATA OURSELF,
# aka x *= RESCALE

# predict(self, x, batch_size=None, verbose=0, steps=None)
x_train *= RESCALE
x_test *= RESCALE

X_bottleneck_train = model.predict(x_train)
X_bottleneck_test = model.predict(x_test)

print("X_bottleneck_train:", X_bottleneck_train.shape)
print("y_test:", y_train.shape)#, y_train[0:10])
print("X_bottleneck_test:", X_bottleneck_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

print("train_data.shape[1:]", X_bottleneck_train.shape[1:])

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=X_bottleneck_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_bottleneck_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_bottleneck_test, y_test),
                    verbose=1)

from visualize_history import visualize_history
visualize_history(history.history, show_also='acc', save=True, save_path='classifier3_'+str(epochs)+'epochs_')
