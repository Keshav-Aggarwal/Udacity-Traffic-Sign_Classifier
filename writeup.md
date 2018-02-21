# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

Please see the Notebook for the Visualizations.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a preprocessing step I shuffled the complete dataset so that Optimizer can work easily. I decided to not change the image data, as image data is very simple and it is 32*32 image. Also color may provide extra information while trying to identify the traffic sign as colors are also part of identifying the traffic sign.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 3x3 stride, VALID padding, outputs 30x30x8 	|
| Convolution 3x3     	| 3x3 stride, VALID padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| DROPOUT     	|  0.8	|
| Convolution 3x3     	| 3x3 stride, VALID padding, outputs 12x12x32 	|
| Convolution 3x3     	| 3x3 stride, VALID padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Convolution 3x3     	| 3x3 stride, VALID padding, outputs 4x4x32 	|
| RELU					|												|
| Fully connected		| Input 4x4x32, output 120        									|
| RELU					|												|
| DROPOUT     	|  0.5	|
| Fully connected		| Input 120, output 84        									|
| RELU					|												|
| Fully connected		| Input 84, output 43        									|
| Softmax				|         									| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
I used the data

| Type         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer         		| AdamOptimizer   							| 
| Batch Size     	| 128 	|
| Epochs     	| 40 	|
| Learning Rate					|			0.0005									|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99
* validation set accuracy of  95
* test set accuracy of 93

If an iterative approach was chosen:
* At first I tried to create a simple CNN with just 1 layer of CNN and 2 FC, but it didn't go well and was not able to perform.
* Then I tried with the LeNet architecture and found that with LeNet I was able to get the accuracy of around 85. So I tried to add more layers.
* I added more Convolution and Fully connected layers and then I was able to get the accuracy of around 94.
* Then I tried to reduce the learning rate to 0.0005 from 0.001 so that accuracy can increase gradually with more epochs.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][Test_Images/30.jpg] 
![alt text][Test_Images/general_caution.jpg] 
![alt text][Test_Images/priority.jpg] 
![alt text][Test_Images/yield.jpg] 
![alt text][Test_Images/stop.jpg]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 30 km/h			| 30 km/h 										|
| Yield					| Road Work											|
| Priority Road 	      		| Priority Road					 				|
| General Caution			| General Caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

#First Image : 'Priority road'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   									| 
| 0.0     				| End of all speed and passing limits 										|
| 0.0					| End of no passing											|
| 0.0	      			| Roundabout mandatory					 				|
| 0.0			    | Vehicles over 3.5 metric tons prohibited      							|

#Second Image : 'Stop'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| Speed limit (60km/h)										|
| .05					| No entry											|
| .04	      			| Speed limit (30km/h)					 				|
| .01				    | Speed limit (80km/h)      							|

#Third Image : '30 Km/hr'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Speed limit (30km/h)   									| 
| .20     				|Speed limit (50km/h) 										|
| .05					| Speed limit (20km/h)											|
| .04	      			| Speed limit (80km/h)					 				|
| .01				    | Speed limit (70km/h)      							|

#Fourth Image : 'General Caution'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| General Caution   									| 
| .20     				|Traffic signals 										|
| .05					| Pedestrians											|
| .04	      			| Road narrows on the right					 				|
| .01				    | Speed limit (70km/h)      							|

#Fifth Image : 'Yield'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Road work   									| 
| .20     				| No passing for vehicles over 3.5 metric tons 										|
| .05					| Dangerous curve to the right											|
| .04	      			| Slippery road					 				|
| .01				    | End of no passing by vehicles over 3.5 metric tons      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Graphed all the filter outputs of all the layers but not able to Understand proerly why First layer is having so much details of the traffic sign. Will look into this and will try to understand this with the help of Trainer.
