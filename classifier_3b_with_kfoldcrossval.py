# SETUP

img_size = None #(20,20)
img_size = (150,150)
epochs = 100
batch_size = 16

# 4-folds: train 2211 , test 737
# 3-folds: train 1965 , test 983
# 10-folds: train 2653 , test 295

k_folds = 3

#(instead of)validation_split = 0.3


RESCALE = 1. / 255 # put data from 0-255 into 0-1
#RESCALE = 0.004
#RESCALE = 0.01
#RESCALE = 0

# GET ALL DATA
# define the classes in here directly

FOLDER = 'saved_images_square_224'
#FOLDER = 'saved_images_square_224_cutandstretchednozeros'
#FOLDER = 'saved_images_square_224_stretched'
#specialname = 'cutnstretchNozeros'
#specialname = 'stretchedNocut'
specialname = '__retest'
#FOLDER = 'saved_images_square_20'

classes_names = ['lp', 'tc', 'tr', 'vt']
num_classes = len(classes_names)
folders = ['data/'+FOLDER+'/train/lp/', 'data/'+FOLDER+'/train/tc/', 'data/'+FOLDER+'/train/tr/', 'data/'+FOLDER+'/train/vt/',
           'data/' + FOLDER + '/test/lp/', 'data/' + FOLDER + '/test/tc/', 'data/' + FOLDER + '/test/tr/',
           'data/' + FOLDER + '/test/vt/',
        ]
labels_texts = classes_names+classes_names
labels = [0, 1, 2, 3, 0, 1, 2, 3]

SHUFFLE_SEED=1337


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
    if SEED is not None:
        random.seed(SEED)

    #sort_order = random.sample(range(len(a)), len(a))

    sort_order = list(range(0,len(a)))
    random.shuffle(sort_order)

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

def split_data(x,y,validation_split=0.2):
    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]

    print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,x_test,y_test

def chunks(l, k):
    ''' Chunk data from list l into k fjords. Not randomizing the order. '''
    # the first chunk may have more data (if len(l) is not divisible by k)
    # if len(l) < k then the last chunks will be empty
    a = np.array_split(np.array(l), k)
    b = []
    for i in a:
        b.append(i.tolist())
    return b

def kfoldcrossval(all_x, all_y, k, shuffle=True):
    # split all_x, all_y into k folds
    # return array of k times [train_x, train_y, test_x, test_y] (test or val)
    # (... arguably this is a bit slow, it would be faster to handle indices)

    # at start we would like to shuffle the two arrays
    if shuffle:
        all_x, all_y = shuffle_two_lists_together(all_x, all_y, SEED=SHUFFLE_SEED)

    # prepare chunks
    all_x_in_chunks = chunks(all_x, k)
    all_y_in_chunks = chunks(all_y, k)

    # now build the array [ [train_x, train_y, test_x, test_y], ... ]
    kfolds_in_array = []

    for i in range(0, k):
        # i-th chunk should be the test data, the rest should serve as train data
        test_x = all_x_in_chunks[i]
        test_y = all_y_in_chunks[i]

        train_x = []
        train_y = []

        for j in range(0,len(all_x_in_chunks)):
            if j != i:
                train_x += all_x_in_chunks[j]
                train_y += all_y_in_chunks[j]

        # np or not np lists omg
        #train_x = [np.asarray(subset) for subset in train_x]
        #train_y = [np.asarray(subset) for subset in train_y]
        #test_x = [np.asarray(subset) for subset in test_x]
        #test_y = [np.asarray(subset) for subset in test_y]

        kfolds_in_array.append([train_x, train_y, test_x, test_y])

    return kfolds_in_array


# GOTTA GO FAST
##X_all_image_data = X_all_image_data[0:100]
##Y_all_labels = Y_all_labels[0:100]

# RESCALE image data, from 0-255 to 0-1
X_all_image_data *= RESCALE

#x = np.asarray(range(0,10))
#y = np.asarray(range(0,10))
#k = 5
#print("x:",x)
#kfolds_in_array = kfoldcrossval(x, y, k)
#for c in kfolds_in_array:
#    print("train",len(c[0]),"=",c[0], "test",len(c[2]),"=",c[2], )

kfolds_in_array = kfoldcrossval(X_all_image_data,Y_all_labels, k_folds)

for fold in kfolds_in_array:
    [x_train, y_train, x_test, y_test] = fold
    print("train", len(y_train), ", test", len(y_test))

print("---")

# NOW REPEAT THIS FOR EVERY FOLD:
#
# (reset model)
# 1.) train model on train_x, train_y for certain number of epoch
#     - you can save history
# 2.) measure accuracy and loss on test_x, test_y
#     -> accuracy_i, loss_i
# (repeat for each fold)
#
# Calculate mean accuracy and mean loss
# Possibly plot all the histories together

ACCURACIES = []
LOSSES = []
HISTORIES = []

for fold in kfolds_in_array:
    [x_train, y_train, x_test, y_test] = fold

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print("x_test:", x_train.shape)
    print("y_test:", y_train.shape)  # , y_train[0:10])
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)  # , y_test[0:10])
    print("-")

    """
    VIZ = False
    if VIZ:
        images = range(0,9)
        for i in images:
            plt.subplot(330 + 1 + i)
            plt.imshow(x_train[i])
        #Show the plot
        plt.show()

    # See how is the distribution in test and train (should be the same)
    VIZ = False # PS: this works only without the categorical data...
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
    """
    # Now for the model

    #continue

    from keras import applications
    model = applications.VGG16(include_top=False, weights='imagenet')

    # HERE WE ACTUALLY HAVE TO EDIT THE DATA OURSELF,
    # predict(self, x, batch_size=None, verbose=0, steps=None)
    X_bottleneck_train = model.predict(x_train)
    X_bottleneck_test = model.predict(x_test)

    print("X_bottleneck_train:", X_bottleneck_train.shape)
    print("y_test:", y_train.shape)#, y_train[0:10])
    print("X_bottleneck_test:", X_bottleneck_test.shape)
    print("y_test:", y_test.shape)#, y_test[0:10])
    print("-")

    print("train_data.shape[1:]", X_bottleneck_train.shape[1:])

    from keras.models import Sequential
    from keras.layers import Dropout, Flatten, Dense

    model = Sequential()
    model.add(Flatten(input_shape=X_bottleneck_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    #model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_bottleneck_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_bottleneck_test, y_test),
                        verbose=0)

    scores = model.evaluate(X_bottleneck_test, y_test, verbose=0)
    loss, accuracy = scores

    print("scores", scores)

    ACCURACIES.append(accuracy)
    LOSSES.append(loss)
    HISTORIES.append(history.history)

print("ACCURACIES", ACCURACIES)
print("LOSSES", LOSSES)

print("Accuracy %.4f%% (+/- %.2f%%)" % (np.mean(ACCURACIES), np.std(ACCURACIES)))
print("Loss %.4f (+/- %.2f%%)" % (np.mean(LOSSES), np.std(LOSSES)))


from visualize_history import visualize_histories, visualize_special_histories
#visualize_histories(HISTORIES, show_also='acc', save=True, save_path='classifier3b_'+str(epochs)+'epochs_'+str(k_folds)+'folds_')

# def visualize_special_histories(histories, plotvalues='loss', show=True, save=False, save_path='', custom_title=None, just_val=False):
visualize_special_histories(HISTORIES, plotvalues='acc', save=True, save_path='classifier3b_'+str(epochs)+'epochs_'+str(k_folds)+'folds_'+specialname)

