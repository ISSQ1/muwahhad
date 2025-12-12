# Hijab Detection model

This is a hijab Detection project


## Requirements


- Python 3.10.7
- OpenCV 4.7.0.72
- Numpy 1.24.2
- keras 2.12.0
- matplotlib 3.7.1
- Pillow 9.4.0
- mediapipe 0.9.2.1


## Main 8 steps of this project :


1. Collecting datasets from Google (Data Crawling).
2. Using mediapipe and opencv, images downloaded from Google were worked on and images containing faces were identified and cropped and saved in a new folder to be given to the model.
3. Using the VGG16 pre-trained network, feature extraction was performed on the images.
4. Making a simple model and combining it with the VGG16 model.
5. Since the number of images to train the model is small, data augmentation was used to augment the training data.
6. The model was trained using the dataset obtained from the faces and the model was saved.
7. The accuracy and loss diagram of the model was saved.
8. The trained and stored model was tested on new images and on the webcam.


## Steps to run this on your system


**Step 1**. To use the VGG16 pre-trained model, you need to download the vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 file.
If you don't have this file or you encounter an error while downloading, this file is located in the vgg16_notop folder.

**Step 2**. To test the model:
You can use the testing-the-model-with-webcam.py file to test the model on your webcam or the testing-the-model-with-new-images.ipynb file to test the model on new images.

