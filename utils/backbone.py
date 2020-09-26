import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util
import time

def set_model(model_name, label_name):

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_SVDMDL = model_name + '/saved_model/saved_model.pb'

	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('Model', label_name)

	num_classes = 1

	# Enable GPU dynamic memory allocation
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
	    tf.config.experimental.set_memory_growth(gpu, True)

	#
	print('Loading model...', end='')
	start_time = time.time()

	# Load saved model and build the detection function
	model = tf.saved_model.load(PATH_TO_SVDMDL)
	detect_fn = model.signatures['serving_default']

	end_time = time.time()
	elapsed_time = end_time - start_time
	print('Done! Took {} seconds'.format(elapsed_time))
	#

	# %%
	# Load label map data (for plotting)
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Label maps correspond index numbers to category names, so that when our convolution network
	# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
	# functions, but anything that returns a dictionary mapping integers to appropriate string labels
	# would be fine.
	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
	                                                                    use_display_name=True)

	return detect_fn, category_index
