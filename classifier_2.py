''' Inspired by blogpost:
"Building powerful image classification models using very little data"
from blog.keras.io.

data in the following structure:
data/*train and validation*/*first and second*/*.jpg
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1456
nb_validation_samples = 816
epochs = 5
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    print("train_data", len(train_data))
    nb_train_samples = len(train_data)

    # fast fix if we had odd number
    if (nb_train_samples % 2)==1:
        train_data = train_data[0:-1]
        nb_train_samples = len(train_data)
        print("Careful, we had odd number of samples in train_data.")

    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    print("train_data", len(validation_data))
    nb_validation_samples = len(validation_data)

    # fast fix if we had odd number
    if (nb_validation_samples % 2)==1:
        validation_data = validation_data[0:-1]
        nb_validation_samples = len(validation_data)
        print("Careful, we had odd number of samples in validation_data.")

    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    print("Mark down this input_shape for the top model: ", train_data.shape[1:])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=2)
    model.save_weights(top_model_weights_path)

    return history

# remember that you can comment the first one to "cook the features" only once
#save_bottlebeck_features()
history = train_top_model()

from visualize_history import visualize_history
visualize_history(history.history, show_also='acc', save=True, save_path='classifier2_'+str(epochs)+'epochs_')