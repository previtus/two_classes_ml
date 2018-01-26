# two_classes_ml
sample for ML working with two class dataset, it's purpose is to have something for fast experiments

# setup and usage

Prepare your dataset, get same size images from two classes and put them into two folders.

- run  the `split.py` with correctly set values for `firstclassfolder, secondclassfolder, train_test_split`
- classifiers are inspired by blog post *"Building powerful image classification models using very little data"* from blog.keras.io (classifier_1) and by *"Using Bottleneck Features for Multi-Class Classification in Keras and TensorFlow"* from codesofinterest.com (classifier_2).

- edit the exact info about your dataset, such as `img_width, img_height, nb_train_samples, nb_validation_samples` and run one of the classifiers
  * `classifier_1.py` - simple CNN model
  * `classifier_2.py` - load model VGG16 pretrained on ImageNet and attach a simple top CNN model
  
- also outputs simple visualization of the loss function over epochs

# performance on an example

As an example I took images from dogs and cats Kaggle dataset, 2000 of examples per class for training, 400 examples per class for validation.

Here are the plots from both classifiers:

![Plot image](https://github.com/previtus/two_classes_ml/blob/master/example_plots/2400catsanddogs-classifier1_100epochs.png)  |  ![Plot image](https://github.com/previtus/two_classes_ml/blob/master/example_plots/2400catsanddogs-classifier2_50epochs.png)
-------------------------------------------------- | --------------------------------------------------
classifier 1 after 100 epochs | classifier 2 after 50 epochs

