"""Define Base class of Saliecy Map"""
import numpy as np
import keras.backend as K 
from abc import ABCMeta, abstractmethod
from six import add_metaclass

@add_metaclass(ABCMeta)
class SaliencyMask(object):
	def __init__(self, model, target_index):
		'''
		Constructor of SaliencyMask
		@param--model-- keras model
		@param--target_index-- the index of node in the last layer to take a gradients on 
		'''
		pass

	def get_mask(self,img):
		'''
		Returns Saliency Mask(unsmoothed) obtained by backprop with recpect to imput img
		@param--img-- image with shape (H,W,C)
		'''
		pass

class GradientSaliencyMask(SaliencyMask):
	'''Gradient Saliency Mask '''
	"""Compute Mask with a vanila gradient"""
	def __init__(self,model,target_index):
		input_placeholder = [model.input]
		grads = model.optimizer.get_gradients(model.output[0][target_index], model.input)
		self._compute_grads = K.function(inputs=input_placeholder,outputs=grads)

	def get_mask(self,img):
		x = np.expand_dims(img, axis=0)
		grads = self._compute_grads([x, 0])[0][0]
		return grads

