
# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/top_5_sign_names_train.png "Top 5 Sign Names Train"
[image2]: ./output_images/top_5_sign_names_valid.png "Top 5 Sign Names Valid"
[image3]: ./output_images/top_5_sign_names_test.png "Top 5 Sign Names Test"
[image4]: ./output_images/countplot_train.png "Countplot Train"
[image5]: ./output_images/countplot_valid.png "Countplot Valid"
[image6]: ./output_images/countplot_test.png "Countplot Test"
[image7]: ./output_images/augmentation_candidates.png "Augmentation Candidates"
[image8]: ./output_images/augment_class.png "Augment Class"
[image9]: ./output_images/brighten_aug.png "Brighten Augmentation"
[image10]: ./output_images/rotate_aug.png "Rotate Augmentation"
[image11]: ./output_images/countplot_aug.png "Countplot Augmentation"
[image12]: ./output_images/tensorboard.png "Tensorboard"
[image13]: ./output_images/training_progress.png "Training Progress"
[image14]: ./output_images/test_images.png "Test Images"
[image15]: ./output_images/test_images_resized.png "Test Images Resized"
[image16]: ./output_images/softmax_test_image_1.png "Softmax test image 1"
[image17]: ./output_images/softmax_test_image_2.png "Softmax test image 2"
[image18]: ./output_images/softmax_test_image_3.png "Softmax test image 3"
[image19]: ./output_images/softmax_test_image_4.png "Softmax test image 4"
[image20]: ./output_images/softmax_test_image_5.png "Softmax test image 5"
[image21]: ./output_images/softmax_test_image_6.png "Softmax test image 6"
[image22]: ./output_images/softmax_test_image_7.png "Softmax test image 7"
[image23]: ./output_images/featuremap_conv2d_0.png "FeatureMap Conv2D 0"
[image24]: ./output_images/featuremap_relu_0.png "FeatureMap Relu 0"
[image25]: ./output_images/featuremap_maxpool_0.png "FeatureMap Maxpool 0"
[image26]: ./output_images/featuremap_pixelated.png "FeatureMap Pixelated"
[image27]: ./output_images/precision.png "Precision"
[image28]: ./output_images/recall.png "Recall"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/samiriff/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32 x 32 x 3`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

I used the Pandas library to analyse the image dataset. After mapping the class IDs to actual sign names (from `signnames.csv`), I found the classes which have the most number of training examples. Then, using the seaborn library, I plotted a countplot to show the distribution of the 43 classes in the training, validation and test sets, and the results are as shown below (Note that only the class IDs are displayed on the X-axes of these plots. Please use `signnames.csv` to find the corresponding sign name):
 - Training Set
 
	![top_5_sign_names_train][image1]
	![countplot_train][image4]
	
 - Validation Set
 
	 ![top_5_sign_names_valid][image2]
	![countplot_valid][image5]
	
 - Test Set 
 
	![top_5_sign_names_test][image3]
	![countplot_test][image6]


As can be observed in all the plots, the data is pretty skewed with plenty of images in a few classes and a dearth of images in others.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to balance the skewed distribution of images in the training set by performing some form of augmentation. 

As part of augmentation, I first had to find suitable candidate image classes for which more data was required. Using the `find_augmentation_candidates()` method, I selected all those image classes for which there were fewer than 1000 images present in the training set. A data frame was populated with these selected candidate image classes along with the count of images and the number of images that had to be generated to bring the count to 1000. A subset of the data frame is as shown below:

![augmentation_candidates][image7]

I defined the `get_augmented_images()` method which returns:
1. An array containing augmented images. 
	- The first half of this array contains images that have been randomly rotated by at most 15 degrees in the left or right direction, using the `rotate()` and `tf.contrib.image.rotate()` method. The figure below depicts the original and the rotated images.
	
	![rotate_aug][image10]
	
	- The second half of this array contains images that have been adjusted in brightness by a random factor using a delta randomly picked in the interval `[-0.3, 0.3]`, using the `brighten()`  and `tf.image.random_brightness()` methods. The figure below depicts the original and the brightened images.
	
	![brighten_aug][image9]
	
2. An array containing the indices of images in the training dataset that were used for augmentation. This particular array is generated using the selected candidate image classes from the previous step, using the `get_augmentation_candidates()` method, which creates an array of the required size, randomly populated with values from a given list of image indices. 

Here is an example of the augmentation step performed on the image class with ID=0 (Speed Limit (20km/h)):

![augment_class][image8]

From the data frame of augmentation candidates, it can be seen that 820 images are required for this class, which means that an array of size `820 x 32 x 32 x 3` is created. The image on the left depicts one of the selected candidate images. The image on the right depicts this image after augmentation. 

Since tensors were being used to augment the images, there was a noticeable performance improvement in generating images when the GPU was enabled.

Shown below is the countplot generated for the training set containing augmented data, resulting in a total of 51690 training examples:

![countplot_aug][image11]

As a last step, I normalized the image data by subtracting `128` from each pixel and dividing the resulting pixel by `128` so that the data has mean zero and equal variance, forming a normal distribution to limit the range of values that will be processed by the neural network. For instance, the mean and standard deviation of the training set before normalization was `82.677` and `67.850` respectively, and after normalization, they are `-0.354` and `0.530` respectively.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6    |
| Bias 6x1				| 							 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  same padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Bias 16x1				| 							 					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  same padding, outputs 14x14x6  	|
| Flatten				| outputs 400									|
| Fully connected		| input 400, outputs 120        				|
| RELU					|												|
| Dropout				| keep_prob 0.75 for training set				|
| Fully connected		| input 120, outputs 84	        				|
| RELU					|												|
| Fully connected		| input 84, outputs 10	        				|
| Softmax 				| outputs probabilities for each image class	|

I used the `tf.softmax_cross_entropy_with_logits()` method to determine the cross-entropy of the output of the model, based on the softmax values and the one-hot-encoding of the training set labels. Then I made use of the Adam Optimizer to minimize the mean of the cross-entropy loss obtained earlier. 

A visual representation of this model based on tensorboard was obtained (with the help of the example at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb) is shown below:
![tensorboard][image12]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameter values:
| Hyperparameter | Value |
|--|--|
| Learning Rate  | 0.001 |
| Number of Epochs | 10  |
| Batch Size |  128      |

I used the Adam Optimizer to reduce the mean of the cross entropy loss function, using the augmented training set of 51,690 training examples I had obtained earlier. I used the `tqdm` library to monitor training progress, by outputing the validation accuracy and loss on the training set after all the batches in each epoch had been processed, as shown below:
![training_progress][image13]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.4%
* test set accuracy of 92.7%

At first, I chose the default LeNet architecture outlined in the lecture videos, and found that the validation accuracy stagnated at around 89-90% after 10 epochs, and didn't vary much even after increasing the number of epochs, although the training accuracy improved, which indicated overfitting. 

Then I tried increasing the batch size to 256 and increasing the number of epochs as well, but the model performed worse than before, with the validation accuracy never going beyond 90%. The training accuracy reached 99.8%, which indicated overfitting.

Increasing the learning rate from 0.001 to 0.01 led to underfitting with a training accuracy of 96.6% and validation accuracy going as low as 86%. Increasing the number of epochs in this case didn't lead to any noticeable improvement because the steps taken during gradient descent were becoming too large due to the higher learning rate. 

After playing around with the hyperparameters, I decided to modify the default LeNet model a bit. Based on the video lectures, I added a new dropout layer to the first fully-connected layer of the network, and I was immediately able to see a marked improvement in the validation accuracy. The dropout layer was preventing overfitting by turning off units in the hidden layer with a probability of 0.25 (or retaining units in the hidden layer with a probability of 0.75). This dropout was applied only on the training set and not the validation and test sets. 

The final validation accuracy I could achieve with this network was 94.4%. For the test set, I evaluated the accuracy as well as the confusion matrix for all labels, using the `get_precision_recall()` method. The accuracy on the test set was 92.7%. 

The precision, which is the ratio of true positives to the sum of true positives and false positives, on the test set for each class is as shown below: 

![precision][image27]

The recall, which is the ratio of true positives to the sum of true positives and false negatives, on the test set for each class is as shown below:

![recall][image28]

The mean precision was 0.9 and the mean recall was 0.89. Details of the image classes for which the precision and recall were minimum or maximum are given in the table below:

| Metric | Min | Max | Image classes with Min | Image classes with Max |
|--|--|--|--|--|
| Precision  | 0.537 | 1.0 | Pedestrians | No entry, Double curve |
| Recall | 0.483 | 1.0 | Pedestrians | Bicycles crossing, Go straight or right|

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image14] 

These images are of different widths and heights, unlike the training dataset which had images of size `32 x 32`.

- The first image contains a "Speed limit (70km/h)" sign which might be difficult to classify because the sign is in front of a blue sky and green trees. 
- The second image contains a blue "turn right ahead" sign against a blue sky, which has a gradient of blue. 
- The third image contains a stop sign and might be difficult to classify because it is a cropped image with a black border. 
- The fourth image contains a "Road work" sign against a noisy background of buildings and clouds in the sky. 
- The fifth image contains a "Road work" sign against an almost-clear blue sky. 
- The sixth image contains a "Speed limit (60km/h)" sign with some written text at the bottom against a blue sky. 
- The sixth image contains a "General caution" sign against a white background with a black band at the bottom.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The first step in classifying this test set was to resize the test images to `32 x 32` to make it compatible with the model:

![test_images_resized][image15]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)	| Stop sign   									| 
| Turn right ahead		| Turn right ahead								|
| Stop 					| Priority Road									|
| Road Work	      		| Road Work						 				|
| Road Work				| Road Work		      							|
| Speed limit (60km/h)  | Speed limit (60km/h)							|
| General Caution  		| General Caution								|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.71%. This compares favorably to the accuracy on the test set of 92.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making determining the top 5 predictions for each test image using my final model is located in the `get_top_k_probabilities()` method.

For the first image, the model is sure that this is a "Speed limit (70km/h)" sign (probability of 1), and the image does contain a "Speed limit (70km/h)" sign. The top five soft max probabilities were:

![softmax_test_image_1][image16]

For the second image, the model is relatively sure that this is a "Turn right ahead" sign (probability of 0.61) and the image does contain a "Turn right ahead" sign. The top five softmax probabilities are:

![softmax_test_image_2][image17]

For the third image, the model is certain that this is a "Priority road" sign (probability of 1) but the image contains a "Stop" sign. The top five softmax probabilities are:

![softmax_test_image_3][image18]

For the fourth image, the model is sure that this is a "Road work" sign (probability of 0.99) and the image does contain a "Road work" sign. The top five softmax probabilities are:

![softmax_test_image_4][image19]

For the fifth image, the model is sure that this is a "Road work" sign (probability of 0.99) and the image does contain a "Road work" sign. The top five softmax probabilities are:

![softmax_test_image_5][image20]

For the sixth image, the model is relatively sure that this is a "Speed limit (60km/h)" sign (probability of 0.97) and the image does contain a "Speed limit (60km/h)" sign. The top five softmax probabilities are:

![softmax_test_image_6][image21]

For the seventh image, the model is sure that this is a "General Caution" sign (probability of 1) and the image does contain a "General Caution" sign. The top five softmax probabilities are:

![softmax_test_image_7][image22]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In order to obtain the names of the activation tensors that had to be passed as parameters to the given `outputFeatureMap()` method, I used the helper methods provided in https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb to generate a visualized graph of my model in tensorboard. It was easy to navigate from inputs and outputs of one operation to another, and obtain the names of the tensors. A test image with the label "Speed limit (70km/h)" was considered.

The first convolutional layer (Conv2D:0) was depicted as 6 different grayscale featuremaps, in which the edges of the number 70 and the circle of the signboard could be discerned:
![featuremap_conv2d_0][image23]

The RELU activation function used on the first convolutional layer (Relu:0) showed a sharper contrasts between the number edges and the circle of the signboard in a few featuremaps:
![featuremap_relu_0][image24]

The MaxPool function used on the Relu activation output above resulted in the following featuremaps containing pixelated data in which the number edges and circle of the signboard could be somewhat seen:
![featuremap_maxpool_0][image25]

The remainder of the section tried to visualize the next convolutional layer and its activation function but I wasn't able to make much sense of the feature maps here since they were highly pixelated:
![featuremap_pixelated][image26]