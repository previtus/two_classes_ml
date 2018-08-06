
img_size = None #(20,20)
img_size = (150,150)
epochs_first = 10
epochs_second = 40
batch_size = 16
validation_split = 0.3
RESCALE = 1. / 255 # put data from 0-255 into 0-1
# GET ALL DATA
# define the classes in here directly

from data_handling import LOAD_DATASET, LOAD_DATASET_VAL_LONGER_THR2, sample_random_subset_from_list, y_from_x
from data_handling import load_images_with_keras,  convert_labels_to_int, convert_back_from_categorical_data, how_many_are_in_each_category

TRAIN_WITH_LONGER_THAN = 1000
TRAIN_C_balanced = 5000

SPLIT = 0.3 # 70% and 30%
FOLDER = 'chillan_saved_images_square_224_ALL_with_len'
folders = ['data/'+FOLDER+'/LP/', 'data/'+FOLDER+'/TR/', 'data/'+FOLDER+'/VT/']

VAL_ONLY_LONGER_THR2 = 1000
BalancedVal = False
StillBalance10to1to1 = True
X_TRAIN_BAL, X_VAL_FULL = LOAD_DATASET_VAL_LONGER_THR2(
    TRAIN_WITH_LONGER_THAN, TRAIN_C_balanced, SPLIT, FOLDER, folders, VAL_ONLY_LONGER_THR2,
    BalancedVal=BalancedVal,StillBalance10to1to1 = StillBalance10to1to1)

specialname = '__Finetuned'

classes_names = ['LP', 'TR', 'VT']
num_classes = len(classes_names)
labels_texts = classes_names
labels = [0, 1, 2]

DROP=0.2
SUBSET_FOR_TRAIN = 8000
SUBSET_FOR_VAL = 8000

############ Whats bellow doesn't have to be changed dramatically

X_TRAIN_BAL,_ = sample_random_subset_from_list(X_TRAIN_BAL, SUBSET_FOR_TRAIN)
Y_TRAIN_BAL = y_from_x(X_TRAIN_BAL)
X_VAL,_ = sample_random_subset_from_list(X_VAL_FULL, SUBSET_FOR_VAL)
Y_VAL = y_from_x(X_VAL)


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras
from matplotlib import pyplot as plt

print("Loading image data!")
# X_TRAIN_BAL, Y_TRAIN_BAL
x_train = load_images_with_keras(X_TRAIN_BAL, target_size=img_size)
y_train = convert_labels_to_int(Y_TRAIN_BAL, classes_names, labels)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

# X_VAL, Y_VAL
x_test = load_images_with_keras(X_VAL, target_size=img_size)
y_test = convert_labels_to_int(Y_VAL, classes_names, labels)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

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

x_train *= RESCALE
x_test *= RESCALE

# =============================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# ROUGH
from keras import optimizers
from keras.applications import VGG16

vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

print("calculating high lvl features...")
X_bottleneck_train = vgg_conv.predict(x_train)
X_bottleneck_test = vgg_conv.predict(x_test)

print("X_bottleneck_train:", X_bottleneck_train.shape)
print("y_test:", y_train.shape)#, y_train[0:10])
print("X_bottleneck_test:", X_bottleneck_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")
print("train_data.shape[1:]", X_bottleneck_train.shape[1:])

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

classifier_model = Sequential()
classifier_model.add(Flatten(input_shape=X_bottleneck_train.shape[1:]))
classifier_model.add(Dense(256, activation='relu'))
classifier_model.add(Dropout(0.5))
classifier_model.add(Dense(num_classes, activation='sigmoid'))

print("FIRST ROUGH MODEL:")
classifier_model.summary()

#classifier_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
classifier_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# ==============================================================================
# TRAIN 1
# ==============================================================================
#

history1 = classifier_model.fit(X_bottleneck_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs_first,
                    validation_data=(X_bottleneck_test, y_test),
                    verbose=1)

# Works well, gets us till cca 96% even in 10 epochs (possibly even 5)

# ==============================================================================
# ==============================================================================

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers

# Create the model
fine_model = models.Sequential()
fine_model.add(vgg_conv)
fine_model.add(classifier_model)

print("SECOND FINE MODEL:")
fine_model.summary()

# Compile the model
# TRY other?

#fine_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
# clip norm didnt help with loss: nan
#fine_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4, clipnorm=1.),metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # default lr lr=0.001

# TRY
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
fine_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


# ==============================================================================
# TRAIN 2
# ==============================================================================
#

history2 = fine_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs_second,
                    validation_data=(x_test, y_test),
                    verbose=1)

# Whoops, sudden drop to loss: nan

# ==============================================================================
# REPORT
# ==============================================================================
#

#print(history1.history)
#print(history2.history)

split_n = len(history1.history['val_loss'])

# val_loss', 'val_acc', 'loss', 'acc
history1.history['val_loss'] += history2.history['val_loss']
history1.history['val_acc'] += history2.history['val_acc']
history1.history['loss'] += history2.history['loss']
history1.history['acc'] += history2.history['acc']

from visualize_history import visualize_history
plt = visualize_history(history1.history, show_also='acc', show=False, save=False)
#visualize_history(history2.history, show_also='acc', save=False, save_path='classifier5b_'+str(epochs)+'epochs_')

plt.axvline(x=split_n-0.5, linestyle='dashed', color='black')

filename = 'classifier5b_CHILL_'+str(epochs_first)+'+'+str(epochs_second)+'epochs_'
plt.savefig(filename)

plt.show()

fine_model.save('5b_final_fine_model.h5')
