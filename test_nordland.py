######################################################################### 
# This script evaluates the place recognition method on the Nordland Dataset
#	It uses the images included in the Nordland test partition and a neural network to extract the features.
#	Features are compared using euclidean distance. Lower distance means higher similarity.
#		Inputs: Deploy file and Caffemodel file paths, Test partition path and index of the compared seasons.  
#		Output: Fraction of correct matches
#   Required: The Caffe Framework must be installed and some python libraries too.

# Created by:
##	Daniel Olid: danielolid94@yahoo.es
##	Jose Maria Facil: jmfacil@unizar.es
##	Javier Civera: jcivera@unizar.es
######################################################################### 

# Imports
import numpy as np
import caffe
from scipy.misc import imread, imresize
import pandas as pd
import argparse
import os
import sys

test_size = 3450
feature_size = 128

######################################################################### 
# Given the input feature, this method finds the closest reference features.
def closestFeatures (input_feature, reference_features):
	distances = np.zeros(reference_features.shape[0]) 
	closest_features_labels = -1*np.ones(reference_features.shape[0])
	reference_features_labels = np.arange(reference_features.shape[0])
	test_size = reference_features.shape[0]
	
	# Calculate euclidean distance in a vectorized way:
	option = 0 # This parameter allows to divide the operation if the vector size is too big
	
	if option == 0:
		distances = np.sum((reference_features - input_feature)**2, axis = 1)
	elif option == 1:
		step = int(test_size/2)
		distances[0:step] = np.sum((reference_features[0:step]-input_feature)**2, axis = 1)
		distances[step:test_size] = np.sum((reference_features[step:test_size]-input_feature)**2, axis = 1)
	elif option == 2:
		distances[0:test_size/4] = np.sum(((reference_features[:,0:test_size/4]-input_feature))**2, axis = 1)
		distances[test_size/4:test_size/2] = np.sum( ((reference_features[:,test_size/4:test_size/2]-input_feature))**2, axis = 1)
		distances[test_size/2:test_size/2+test_size/4] = np.sum(((reference_features[:,test_size/2:test_size/2+test_size/4]-input_feature))**2, axis = 1)
		distances[test_size/2+test_size/4:test_size] = np.sum(((reference_features[:,test_size/2+test_size/4:test_size]-input_feature))**2, axis = 1)

	# We order the distances in a dataframe
	distance_frame = pd.DataFrame(data={"dist": distances, "idx": reference_features_labels})
	if sys.version_info[0] < 3:
		# Python 2.X
		distance_frame.sort_values("dist", inplace=True)
	else:
		# Python 3.X
		distance_frame.sort("dist", inplace=True)
	# We return the labels of the closest features. Since they are synced, the labels correspond to the places index.
	closest_features_labels = distance_frame.iloc[:]["idx"].values
	#print("Lower distance: ", (distance_frame.iloc[:]["dist"].values)[0])
	#print("Higher distance: ", (distance_frame.iloc[:]["dist"].values)[999])
	return closest_features_labels

######################################################################### 
# Given the images and the neural network, this method extracts the features.
# 	Inputs: Path to the images, number of images, season index and neural network.
#	Output: Feature matrix with dimension: [number of images, feature size]
#           Label vector with dimension: [number of images]
def extractFeatures(n_images, season, net,path_im):
	features = []
	labels = []
	count = 0
	mean = np.array([104.968053883,  119.316329094, 112.631406523]) # Nordland mean (bgr)
	#mean = np.array([103.939, 116.779, 123.68]) # Imagenet mean (bgr)
	#mean = np.array([105.487823486, 113.741088867, 116.060394287]) # Places mean (bgr)
	while count < n_images:
		image, label = loadImage(count, season,path_im)
		image_bgr = np.zeros((1,3,224, 224)) # Input image colour channel order is bgr
		image_bgr[0, 0, :, :] = image[:, :, 2] - mean[0]
		image_bgr[0, 1, :, :] = image[:, :, 1] - mean[1]
		image_bgr[0, 2, :, :] = image[:, :, 0] - mean[2]
		out = net.forward_all(data = np.asarray([image_bgr]))
		feat = net.blobs['feat'].data.copy() # If another output layer is used, name 'feat' should be changed
		feat = np.reshape(feat, (feature_size))
		features.append(feat)
		labels.append(count)
		if ( count % 1000 == 0):
			print(count,' features extracted...')
		count = count + 1
	print(count,' features extracted.')
	return (np.asarray(features), np.asarray(labels))
	
######################################################################### 
# Given an index and a season, this method loads an image
def loadImage ( image_index, season ,path_im):
	path = path_im[season]+'section1/'+str(image_index)+'.png'
	if image_index >= 1150 and image_index < 1150*2:
		path = path_im[season]+'section2/'+str(image_index)+'.png'
	elif image_index >= 1150*2:
		path = path_im[season]+'section3/'+str(image_index)+'.png'
	image = imread(path)
	label = image_index
	return (image, label)

######################################################################### 
# Given the closest features, this method counts the number of correct matches
#    The number of neighbours parameter can be explained with an example. 
#    This method works in an iterative way. 
#    If the "number_of_neighbours" is 3, the procedure will be the following: 
#       First iteration: This method checks if the closest match is correct.
#       Second iteration: This method checks how many of the two closest places are correct.
#       Third iteration: This method checks how many of the three closest places are correct.
#    The method will output the number of correct matches for each iteration.
def numberOfCorrectMatches(number_of_neighbours, closest_labels, real_label):
    correct_votes = 0
    for i in range(number_of_neighbours):
        if (closest_labels[i] >= real_label-2 and closest_labels[i] <= real_label+2):
            correct_votes = correct_votes + 1
    return correct_votes
	
###############################################################################################
######################################### MAIN ################################################
###############################################################################################

def main():
	parser = argparse.ArgumentParser(description='Test Single-View Place Recognition.')
	parser.add_argument('--dataset',required=True,type=str,help="Path to Partitioned Nordland. e.g. for given/path/ it should contain given/path/test ")
	parser.add_argument('--input_season',default=0,type=int,required = False, help="[default = 0] 0 = summer, 1 = winter, 2 = fall, 3 = spring.")
	parser.add_argument('--reference_season',default=1,type=int,required = False, help="[default = 1] 0 = summer, 1 = winter, 2 = fall, 3 = spring.")
	parser.add_argument('--gpu',default=1,type=int,help="[default =  1] 0 = cpu, 1 = gpu")
	parser.add_argument('--model',required=False,
		default='./models/vgg16_pool4_fc128_deploy.prototxt',type=str,
		help="Paths to the deploy file, default= ./models/vgg16_pool4_fc128_deploy.prototxt")
	parser.add_argument('--params',required=False,
		default='./models/fine_tuned/partitioned_norland_fine_tuned.caffemodel',
		type=str,help="Paths to the caffemodel file, default= ./models/fine_tuned/partitioned_norland_fine_tuned.caffemodel")
	args = parser.parse_args()
	# Seasons to compare. Change as desired.
	# First season is the input one. Second season is the reference season.
	input_season = args.input_season     # 0 = summer, 1 = winter, 2 = fall, 3 = spring.
	reference_season = args.reference_season # 0 = summer, 1 = winter, 2 = fall, 3 = spring.

	# We use Caffe in GPU mode. It can be changed but the algorithm will run slower. 
	if args.gpu == 1:
		caffe.set_mode_gpu()

	# Declare here the paths to the deploy, caffemodel and the images.
	# I prefer absolut paths, change them if you want. If you use this paths, please make sure to put everything in the corresponding folder.
	deploy_prototxt_file_path = args.model # Our article network deploy

	# caffe_model_file_path = 'models//not_fine_tuned//triplet_vgg16_pool4_fc_128_iter_208680.caffemodel' # Our non fine tuned model
	caffe_model_file_path = args.params # Our fine tuned model

	path_im = [ os.path.join(args.dataset,sp) for sp in ['test/summer_images_test/', 
		'test/winter_images_test/',
		'test/fall_images_test/',
		'test/spring_images_test/']]
	
		
	######################################################################### 
	# The rest of the parameters shouldn't be changed unless you want to experiment with other networks/datsets or change the algorithm.
	

	# CNN reconstruction and loading the trained weights
	net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

	# Variables
	count = 0
	num_neighbours = 5 # Number of neighbours
	precision_at_k = np.zeros(num_neighbours) # Fraction of correct matches
	at_least_one_at_k = np.zeros(num_neighbours) # Fraction of matches with at least one correct place
	closest_places = np.zeros((test_size, num_neighbours)) # Matrix to save the closest features labels
	all_matches = np.zeros((test_size, test_size)) # Matrix to save all the features labels ordered by distance

	# Feature extraction
	print(" Extracting features...this may take a while...")
	input_features, input_labels = extractFeatures(test_size, input_season, net,path_im) # Dims: [number of images, feature size], [number of images]
	reference_features, reference_labels = extractFeatures(test_size, reference_season, net,path_im)
	print(" Features extracted...")
	print(" Initializing comparison...")

	for i in range(test_size):
		#print(" Input place: ", input_labels[i]) 
		closest_places_labels = closestFeatures(input_features[i], reference_features)
		closest_places[i,:] = closest_places_labels[0:num_neighbours]
		all_matches [i,:] = closest_places_labels
		#print(" Closest reference places: ", reference_labels[closest_places_labels[0:num_neighbours]])
		for j in range(num_neighbours):
			number_of_votes = numberOfCorrectMatches(j+1, reference_labels[closest_places_labels[0:num_neighbours]], input_labels[i])
			precision_at_k [j] += number_of_votes
			if ( number_of_votes >= 1 ):
				at_least_one_at_k [j] += 1

	# Output the metrics				
	print("#"*20)
	print("Evaluated Neural network: ", caffe_model_file_path)
	string_seasons = ["summer", "winter", "fall", "spring"]
	print("Input season: ", string_seasons[input_season])
	print("Reference season: ", string_seasons[reference_season])

	# Fraction of correct matches 
	precision_at_k = precision_at_k/test_size
	at_least_one_at_k = at_least_one_at_k/test_size
	# Percentages are made by considering all the input places
	for neighbour in range(len(precision_at_k )):
		precision_at_k [neighbour] = precision_at_k[neighbour]/(neighbour+1)
		
	print("Fraction of correct matches: ", precision_at_k[0]*100.0, "%")
	"""
	print(" Fraction of correct matches (considering 1 to 5 closest neighbours) is: ")
	print(precision_at_k)
	print(" Fraction of matches (considering 1 to 5 closest neighbours) with at least one correct match in them: ")
	print(at_least_one_at_k)
	"""
	print("")


if __name__ == "__main__":
    main()