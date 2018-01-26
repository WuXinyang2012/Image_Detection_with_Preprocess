This directory contains all the scripts and networks used to detect and classify fruits and vegetables from images taken in Hochschule der Medien(HdM)'s Smart Kitchen system.

In this diretory, there are 10 images named from test1.jpg to test10.jpg which are used to test the accuracy of the whole pipeline.

And also two networks: one is Google's official Inception-V3 network, which is in sub-directory Inception-v3; Another one is our own retrained model, which is in sub-directory Retrained_model.

And also some scripts to do the detection and classification:

1.) KMeans_Watershed.py
In this module, K-means clustering algorithm and Watershed algorithm method are combined, along with some other common image processing algorithm, to pre-process the test images. The method offered in this module will detect how many kinds of fruits or vegetables in each test image and then divide the test image in some subimages, each of which contains only one kind of fruits or vegetables, and store the subimages into one sub-directory named as [Image Name]_subdir.

2.)KW_parameters.py
This script offers a visualized result to enalbe users to fine-tune the parameters of the algorithms used in above mentiond KMeans_Watershed module.
The main parameters which may be tuned to affect the accuracy are threshold values of the pyramid mean-shiflt filter, located in line 38, and the kernel of morphological operations, located in line 97. 
After you get better parameters, remember also tune them in module KMeans_Watershed.

3.)run_retrained.py
Usage: 
$python run_retrained.py [Image Name]

For example: 
$python run_retrained.py test1.jpg

This script will run the retrained model to classify desired images. Besides the classification result, it will output two images: K-means clustered images named as KMeans_[Image Name], and detected images named as output_[Image Name], and a sub-directory named as [Image Name]_subdir. 

4.)run_Inception.py
Usage: 
$python run_Inception.py [Image Name]

For example: 
$python run_Inception.py test1.jpg

This script will run the Google's official model to classify desired images. Besides the classification result, it will output two images: K-means clustered images named as KMeans_[Image Name], and detected images named as output_[Image Name], and a sub-directory named as [Image Name]_subdir. 

Notation: This script will run networks in Intel's Neural Compute Stick. Make sure you have it and already set up SDK.

5.)label_image.py
Usage:
$python label_image.py [Image Name]

For example: 
$python label_image.py test1.jpg

This script will directly run our retrained networks to classify the images without any image pre-processing.
