from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Mimic-III visualization
groups = [
	[[0, 1], 59],
	[[2], 60],
	[[3], 61],
	[np.arange(4,12), 62],
	[np.arange(12,24), 63],
	[np.arange(24,37), 64],
	[np.arange(37,49), 65],
	[[49], 66],
	[[50], 67],
	[[51], 68],
	[[52], 69],
	[[53], 70],
	[[54], 71],
	[[55], 72],
	[[56], 73],
	[[57], 74],
	[[58], 75],
]
def visualize(matrix, title, agg='argmax', vmin=None, vmax=None,save=False, cmap='seismic',img_name='temp'):
	fig = plt.figure(figsize = (17,9))
	ax = plt.gca()	
	to_show = np.zeros((len(groups), 48))
	yticklabels = []
	for i, (row_indices, mask_index) in enumerate(groups):
		if len(row_indices) == 1:
			row = matrix[row_indices[0]]
		else:
			if agg == 'argmax':
				row = matrix[row_indices].argmax(axis=0) / len(row_indices)
			elif agg == 'sum':
				row = matrix[row_indices].sum(axis=0)
		#mask_row = matrix[mask_index]
		to_show[i] = row
		#to_show[2*i+1] = mask_row	
		name = discretizer_header[row_indices[0]].split('->')[0]
		yticklabels.append(name)
		#yticklabels.append("Mask for {}".format(name))
	
	ax.tick_params(axis='both', which='major', labelsize=14, length=7)
	im = ax.imshow(to_show, cmap=cmap, vmin=vmin, vmax=vmax)
	plt.title(title)
	plt.xlabel('Time Axis')
	plt.tick_params(labeltop=False, labelright=True)
	plt.yticks(range(len(yticklabels)), yticklabels)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="5%", pad=0.6)
	plt.colorbar(im,cax=cax,orientation='horizontal')
	if save:
		fig.tight_layout()
		fig.savefig(img_name, dpi=200)

VGG_INPUT_SIZE = 224
def load_image(img_path) :
	'''Load image'''
	image = load_img(img_path, target_size=(VGG_INPUT_SIZE, VGG_INPUT_SIZE))
	image = img_to_array(image)

	return image

def save_image( npdata, path):
	'''Save image'''
	img = Image.fromarray(np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save(path)
	

def show_image(image, grayscale = True, ax=None, title=''):
	if ax is None:
		plt.figure()
	plt.axis('off')
	
	if len(image.shape) == 2 or grayscale == True:
		if len(image.shape) == 3:
			image = np.sum(np.abs(image), axis=2)
			
		vmax = np.percentile(image, 99)
		vmin = np.min(image)

		plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
		plt.title(title)
	else:
		image = image #+ 127.5
		image = image.astype('uint8')
		
		plt.imshow(image)
		plt.title(title)