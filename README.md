# two_classes_ml
sample for ML working with two class dataset, it's purpose is to have something for fast experiments

# setup and usage

Prepare your dataset, get same size images from two classes and put them into two folders.

- run  the `split.py` with correctly set values for `firstclassfolder, secondclassfolder, train_test_split=0.3`
- classifiers are inspired by blog post *"Building powerful image classification models using very little data"* from blog.keras.io
- edit the exact info about your dataset, such as `img_width, img_height, nb_train_samples, nb_validation_samples` and run one of the classifiers
  * `classifier_1.py` - simple CNN model
  * `classifier_2.py` - load model VGG16 pretrained on ImageNet and attach a simple top CNN model
  * `classifier_3.py` - load pretrained VGG16 and pretrained top model and then finetune both (change where you want to freeze the model)
  
- also outputs simple visualization of the loss function over epochs
