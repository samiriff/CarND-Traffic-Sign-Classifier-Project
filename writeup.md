
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

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


