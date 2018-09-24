# Multidigit-Number-Classifier
Keras implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) paper from Google Street View and reCAPTCHA Teams
-----------------------------

Instruction:
-----------------------------
1) download SVHN dataset format 1 (train,test) from http://ufldl.stanford.edu/housenumbers/
train.tar.gz : http://ufldl.stanford.edu/housenumbers/train.tar.gz
test.tar.gz : http://ufldl.stanford.edu/housenumbers/test.tar.gz

2) extract images from train.tar.gz to ./train/train/
	and from test.tar.gz to ./test/test/
   the digitstruct.mat should be in the same location as images

3) run digitstruct_to_npy.ipynb to get 
	bbox_data.npy and labels.npy for train set
	test_bbox_data.npy and test_labels.npy for test set

4) create folder train/croppedtight/ , train/croppedexpand/ , train/croppedsampled/ ,
	and test/croppedtight/ 
	as location to store processed image

5) run sample_from_data.ipynb to generate samples from data for training and testing

6) run model_learning_rate, model_conv_kernel_size, model_activation ipynb files
	to start training models with different parameters and see the result validation accuracy

7) run model_final.ipynb to train final model to see the result validation accuracy
	and save weights into final_model_weight.npy

8) run final_model_eval.ipynb to see the result testing accuracy using test set and 
	see some misclassified result
  
-----------------------------



File description:
-----------------------------
ipynb files:

	digitstruct_to_npy.ipynb - convert digitstruct.mat into .npy files
		bbox_data.npy and labels.npy for train set
		test_bbox_data.npy and test_labels.npy for test set

	sample_from_data.ipynb - process images in the dataset for training and testing
		using .npy files from digitstruct_to_npy.ipynb, store processed image in
		specified path 
		(train/croppedtight/ , train/croppedexpand/, train/croppedsampled/ , test/croppedtight/) 
	
	model_learning_rate.ipynb - start with initial model (lr = 0.0001) and tune only learning_rate to see
		the difference in validation accuracy (for lr = 0.001, 0.0005, 0.00005) (epoch = 20)

	model_conv_kernel_size.ipynb - start with initial model (kernel_size = 3) and tune only kernel_size
		to see the difference in validation accuracy (for kernel_size = 5, 7, 9) (epoch = 20)
	
	model_activation.ipynb - start with initial model (activation = relu) and tune only activation
		function to see the difference in validation accuracy (for activation = tanh, sigmoid) (epoch = 20)
	
	model_final.ipynb - modify parameter from above model (lr = 0.00005, kernel_size = 7,activation = tanh)
		to train final model (epoch = 50), model weight will be stored in final_model_weight.npy

	final_model_eval.ipynb - use test set to check testing accuracy and misclassified samples
-----------------------------
npy files:
	
	bbox_data.npy - store bounding box data of train set from digitstruct.mat
	
	labels.npy - store label data of train set from digitstruct.mat

	test_bbox_data.npy - store bounding box data of test set from digitstruct.mat
	
	test_labels.npy - store label data of test set from digitstruct.mat
	
	final_model_weight.npy - store weights in the trained final model (structure in model_final.ipynb)
-----------------------------
data (not included, please download dataset according to instructions below):

	train/train/*.png - train set (the same in the SVHN dataset format 1 available online)

	train/train/digitstruct.mat - file containing digit information(counding box, label) 
		of the train set (the same in the SVHN dataset format 1 available online)
	
	test/test/*.png - test set (the same in the SVHN dataset format 1 available online)

	test/test/digitstruct.mat - file containing digit information(counding box, label) 
		of the test set (the same in the SVHN dataset format 1 available online)
-----------------------------
processed data (not included, please download and process dataset to produce 
		processed data according to instructions below):
	
	train/croppedtight/*.png - crop region containing all digits in the images in train set,
		rescale to 54x54
	
	train/croppedexpand/*.png - find region containing all digits in the images in train set,
		then expand the size in both x, y direction by 30%, crop and rescale to 54x54

	train/croppedsampled/*.png - find region containing all digits in the images in train set,
		then expand the size in both x, y direction by 30%, crop and rescale to 64x64, 
		random sample 54x54 images (5 samples) from original image
		*** this is the training set and validation set of the model

	test/croppedtight/*.png - crop region containing all digits in the images in test set,
		rescale to 54x54
		*** this is the testing set of the final model
 
-----------------------------
  	summary.pdf - summary of the result of the project 
