from __future__ import absolute_import
from __future__ import print_function


from keras.layers import LSTMCell
from keras import backend as K
import numpy as np 
from numpy import newaxis as na

from keras.layers import Input
from keras.layers import LSTMCell,RNN,Lambda,Dense,Bidirectional,Add,LSTM,Layer,Concatenate,Masking,Dropout,InputLayer
from keras import Model
from keras.utils.generic_utils import CustomObjectScope
import pickle

import time, sys
def update_progress(job_title, progress, length=20):
	block = int(round(length*progress))
	msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
	if progress >= 1: msg += " DONE\r\n"
	sys.stdout.write(msg)
	sys.stdout.flush()

def lrp_viz(input_data, file_name, model):
	
	all_activations = {}
	#Slice layers
	for slice_layer_i in range(2,2+17):
		slice_layer = model.layers[slice_layer_i]
		all_activations[slice_layer.name] = get_target_activation(model.input,slice_layer.output,input_data)[0]
		update_progress("Slice layers", (slice_layer_i-2)/17)
	update_progress("Slice layers", 1)
	
	#bi_lstm_layers
	for bi_lstm_layer_i in range(2+17,2+17+17):
		bi_lstm_layer = model.layers[bi_lstm_layer_i]
		all_activations[bi_lstm_layer.name] = cashe_lstm_forward_activations_batch(bi_lstm_layer,all_activations['slice_'+str(bi_lstm_layer_i-18)],bidirectional=True,dim=8//2)
		print(bi_lstm_layer.name)
		update_progress("Bi_lstm layers", (bi_lstm_layer_i-2-17)/17)
	update_progress("Bi_lstm layers", 1)
	
	#Concat Layer
	concat_layer = model.layers[2+17+17]
	bid_outs = []
	for i in range(1,18):
		bid = all_activations['bidirectional_'+str(i+17)]
		bid_outs.append(np.concatenate([bid['h_Left'][:,:-1],bid['h_Right'][:,:-1][:,::-1]],axis = -1))
		update_progress("Concat Layer", (i-1)/17)
	  
	all_activations[concat_layer.name] = np.concatenate(bid_outs,axis = -1)
	update_progress("Concat Layer", 1)
	
	#LSTM Layer
	last_lstm_layer = model.layers[2+17+17+1]
	all_activations[last_lstm_layer.name] = cashe_lstm_forward_activations_batch(last_lstm_layer,all_activations[concat_layer.name],bidirectional=False,dim=32)
	update_progress("LSTM Layer", 1)
	
	#Dense Layer
	dense_layer = model.layers[2+17+17+3]
	all_activations[dense_layer.name] = get_target_activation(dense_layer.input, dense_layer.output,all_activations[last_lstm_layer.name]['h_Left'][:,-2])[0]
	update_progress("Dense Layer", 1)
	
	
	with open(file_name, 'wb') as handle:
		pickle.dump(all_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_target_activation(input_t, target_t, data):
	"""
	#Inputs
		@input_t --- [tensor or placeholder]
		@target_t --- [tensor or placeholder]
		#data --- [numpy]it must fed into input_t and pass to target_t
	#Returns
		[numpy]target_t activation
	"""
	if not isinstance(target_t,list):
		target_t = [target_t]
	func = K.function([input_t, K.learning_phase()], target_t)
	return [item.astype(np.float64) for item in func([data,1])]

def cashe_lstm_forward_activations_batch(lstm_layer,batch_data,bidirectional,dim):
	b = batch_data.shape[0]
	T = batch_data.shape[1]
	d = dim
	
	gates_pre_Left = np.zeros((b, T, 4*d))  # gates i, f, c, o pre-activation
	gates_Left     = np.zeros((b, T, 4*d))  # gates i, f, c, o activation
	h_Left         = np.zeros((b, T+1, d))
	c_Left         = np.zeros((b, T+1, d))
	
	if bidirectional:
		gates_pre_Right = np.zeros((b, T, 4*d))  # gates i, f, c, o pre-activation
		gates_Right     = np.zeros((b, T, 4*d))  # gates i, f, c, o activation
		h_Right         = np.zeros((b, T+1, d))
		c_Right         = np.zeros((b, T+1, d))
	
	
	for t in range(T):
		batch_t = batch_data[:,:(t+1),:]
		
		if not bidirectional:
			l_cell = lstm_layer.cell
		else:
			l_cell = lstm_layer.forward_layer.cell
		
		gates_and_pre_gates = [l_cell.pre_gates,l_cell.gates,l_cell.h,l_cell.c]
		activations = get_target_activation(lstm_layer.input,gates_and_pre_gates,batch_t)
		
		gates_pre_Left[:,t,:] = activations[0].copy()
		gates_Left[:,t,:]     = activations[1].copy() 
		h_Left[:,t] = activations[2].copy()#h
		c_Left[:,t] = activations[3].copy()#c
		
		if bidirectional:
			batch_t_rev = batch_data[:,T-t-1:,:]

			r_cell = lstm_layer.backward_layer.cell
			gates_and_pre_gates = [r_cell.pre_gates,r_cell.gates,r_cell.h,r_cell.c]
			activations = get_target_activation(lstm_layer.input,gates_and_pre_gates,batch_t_rev)
		
			gates_pre_Right[:,t,:] = activations[0].copy()
			gates_Right[:,t,:]     = activations[1].copy() #i
			h_Right[:,t] = activations[2].copy()#h
			c_Right[:,t] = activations[3].copy()#c
	
	s = np.concatenate([h_Left[:,-2],h_Right[:,-2]],axis = -1) if bidirectional else h_Left[:,-2]
	left = {"gates_pre_Left" : gates_pre_Left,
			"gates_Left":gates_Left,
			"h_Left":h_Left,
			"c_Left":c_Left,
			'final':s
		   }

	if bidirectional:
		right = {"gates_pre_Right" : gates_pre_Right,
		"gates_Right":gates_Right,
		"h_Right":h_Right,
		"c_Right":c_Right
		}
		return {**left,**right}
	return left

def lrp_dense_batch(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor,layer_name, debug=False):
	"""
	LRP for a linear layer with input dim D and output dim M.
	Args:
	- hin:            forward pass input, of shape (batch_size,D,)
	- w:              connection weights, of shape (batch_size, M)
	- b:              biases, of shape (M,)
	- hout:           forward pass output, of shape (batch_size,M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
	- Rout:           relevance at layer output, of shape (b,M,)
	- bias_nb_units:  number of lower-layer units onto which the bias/stabilizer contribution is redistributed
	- eps:            stabilizer (small positive number)
	- bias_factor:    for global relevance conservation set to 1.0, otherwise 0.0 to ignore bias redistribution
	- layer_name      str
	Returns:
	- Rin:            relevance at layer input, of shape (batch_size,D,)
	"""

	sign_out = np.where(hout[:,:]>=0, 1., -1.)[:,na,:] # shape (b, M)
	numer    = np.vstack(([(w * hin[i,:,na])[na,] for i in range(hin.shape[0])])) + ((bias_factor*b[na,:]*1. + eps*sign_out*1.) * 1./bias_nb_units ) # shape (D, M)
	denom    = hout[:,na,:] + (eps*sign_out*1.)   # shape (b, M)
	message  = (numer/denom) * Rout[:,na,:]       # shape (b, D, M)
	Rin      = message.sum(axis=-1)              # shape (D,)
	# Note: local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D
	#       global network relevance conservation if bias_factor==1.0 (can be used for sanity check)
	if debug:
		print("{} local diff: ".format(layer_name), Rout.sum() - Rin.sum())
	return Rin

def lrp_concat(hin_shapes,axis, Rout, layer_name, debug=False):
	"""
	#inputs
	  @hin  --- [list]
	  @hout --- [numpy] concatanetion of hin
	  @axis --- [int]concatantion axis
	  @Rout --- [numpy]relevance score of hout
	  @layer_name --- [str]
	#Outputs
	  @Rin  --- [list] (len(hin) == len(Rin))
	"""
	sections = [item[axis] for item in hin_shapes]
	for i in range(1,len(hin_shapes)):
		sections[i] += sections[i-1]

	if Rout.shape[axis] == sections[-1]:
		sections = sections[:-1]
	Rin = np.split(Rout,sections,axis=axis)
	if debug:
		print("{} local diff: ".format(layer_name), Rout.sum() - np.sum(Rin))
	return Rin
  
##Slice layers LRP
def lrp_slice(hin_shape, Rout,indices,layer_name, debug=False):
	"""
	#inputs
	  @hin  --- [list]
	  @hout --- [numpy] concatanetion of hin
	  @Rout --- [numpy]relevance score of hout
	  @indices --- [list]slicing indecies
	  @layer_name --- [str]
	#Outputs
	  @Rin  --- [list] (len(hin) == len(Rin))
	"""
	Rin = np.zeros(hin_shape)
	Rin[...,indices] = Rout.copy()
	if debug:
		print("{} local diff: ".format(layer_name), Rout.sum() - Rin.sum())
	return Rin

def lrp_lstm_batch(activations,hin, Rout_Left_last, batch_size, seq_len, hidden_dim, input_dim, lstm_weights, bidiractional, layer_name, Rout_Right_last=None, eps=0.001, bias_factor=0, debug=False):
	"""
	Update the hidden layer relevances by performing LRP for the target class LRP_class
	hin --- (batch, timestamp, input_dim)
	"""
	# forward pass
	if Rout_Right_last is None and bidiractional:
		raise('need to specify Rh_Right_last if bidiractional')
	
	activations_ = activations[layer_name]
	gates_pre_Left = activations_['gates_pre_Left']
	gates_Left = activations_['gates_Left']
	h_Left = activations_['h_Left']
	c_Left = activations_['c_Left']
	s = activations_['final']
	
	if bidiractional:
		gates_pre_Left = activations_['gates_pre_Left']
		gates_Left = activations_['gates_Left']
		h_Left = activations_['h_Left']
		c_Left = activations_['c_Left']
		gates_pre_Right = activations_['gates_pre_Right']
		gates_Right = activations_['gates_Right']
		h_Right = activations_['h_Right']
		c_Right = activations_['c_Right']
		s = activations_['final']    
	  
	#T = data.shape[1] seq_len
	#d = 60            hidden_dim
	
	#e      = 60       input_dim
	#C      = 5  # number of classes
	#idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
	
	# initialize
	Rx       = np.zeros((batch_size,seq_len,input_dim))
	Rx_rev   = np.zeros((batch_size,seq_len,input_dim))
	
	Rh_Left  = np.zeros((batch_size,seq_len+1, hidden_dim))
	Rh_Left[:,seq_len-1] = Rout_Left_last.copy()
	Rc_Left  = np.zeros((batch_size,seq_len+1, hidden_dim))
	Rg_Left  = np.zeros((batch_size,seq_len,   hidden_dim)) # gate g only
	
	
	Wxh_Left = lstm_weights[0].transpose()
	Whh_Left = lstm_weights[1].transpose()
	b_Left = lstm_weights[2]
	
	if bidiractional:
		Rh_Right = np.zeros((batch_size,seq_len+1, hidden_dim))
		Rh_Right[:,seq_len-1] = Rout_Right_last.copy()
		Rc_Right = np.zeros((batch_size,seq_len+1, hidden_dim))
		Rg_Right = np.zeros((batch_size,seq_len,   hidden_dim)) # gate g only
		
		Wxh_Right = lstm_weights[3].transpose()
		Whh_Right = lstm_weights[4].transpose()
		b_Right = lstm_weights[5]
		
		hin_rev = hin[:,::-1,:].copy()


	T,d,e = seq_len,hidden_dim,input_dim
	for t in reversed(range(T)):
		x_t = hin[:,t,:].reshape(batch_size,e)
		Rc_Left[:,t]   += Rh_Left[:,t]
		Rc_Left[:,t-1]  = lrp_dense_batch(gates_Left[:,t,d:2*d]*c_Left[:,t-1],         np.identity(d), np.zeros((d)), c_Left[:,t], Rc_Left[:,t], 2*d, eps, bias_factor, layer_name=None, debug=False)
		Rg_Left[:,t]    = lrp_dense_batch(gates_Left[:,t,0:d]*gates_Left[:,t,2*d:3*d], np.identity(d), np.zeros((d)), c_Left[:,t], Rc_Left[:,t], 2*d, eps, bias_factor, layer_name=None,debug=False)
		Rx[:,t]         = lrp_dense_batch(x_t,        Wxh_Left[2*d:3*d].T, b_Left[2*d:3*d], gates_pre_Left[:,t,2*d:3*d], Rg_Left[:,t], d+e, eps, bias_factor, layer_name=None, debug=False)
		Rh_Left[:,t-1]  = lrp_dense_batch(h_Left[:,t-1],Whh_Left[2*d:3*d].T, b_Left[2*d:3*d], gates_pre_Left[:,t,2*d:3*d], Rg_Left[:,t], d+e, eps, bias_factor, layer_name=None, debug=False)
		
		if bidiractional:
			x_t_rev = hin_rev[:,t,:].reshape(batch_size,e)
			Rc_Right[:,t]  += Rh_Right[:,t]
			Rc_Right[:,t-1] = lrp_dense_batch(gates_Right[:,t,d:2*d]* c_Right[:,t-1],         np.identity(d), np.zeros((d)),c_Right[:,t], Rc_Right[:,t], 2*d, eps, bias_factor,layer_name=None, debug=False)
			Rg_Right[:,t]   = lrp_dense_batch(gates_Right[:,t,0:d]* gates_Right[:,t,2*d:3*d], np.identity(d), np.zeros((d)),c_Right[:,t], Rc_Right[:,t], 2*d, eps, bias_factor,layer_name=None, debug=False)
			Rx_rev[:,t]     = lrp_dense_batch(x_t_rev,        Wxh_Right[2*d:3*d].T, b_Right[2*d:3*d], gates_pre_Right[:,t,2*d:3*d], Rg_Right[:,t], d+e, eps, bias_factor, layer_name=None, debug=False)
			Rh_Right[:,t-1] = lrp_dense_batch(h_Right[:,t-1],   Whh_Right[2*d:3*d].T, b_Right[2*d:3*d], gates_pre_Right[:,t,2*d:3*d], Rg_Right[:,t], d+e, eps, bias_factor, layer_name=None, debug=False)
	
	
	Rin = Rx + Rx_rev[:,::-1,:] if bidiractional else Rx
	Rout  = Rout_Left_last+Rout_Right_last[:,::-1] if bidiractional else Rout_Left_last
	if debug:
		print("{} local diff: ".format(layer_name), Rout.sum() - Rin.sum())
	return Rin
	
class LRP_LSTMCell(LSTMCell):
	def call(self, inputs, states, training=None):
		if 0 < self.dropout < 1 and self._dropout_mask is None:
			self._dropout_mask = _generate_dropout_mask(
				K.ones_like(inputs),
				self.dropout,
				training=training,
				count=4)
		if (0 < self.recurrent_dropout < 1 and
				self._recurrent_dropout_mask is None):
			self._recurrent_dropout_mask = _generate_dropout_mask(
				K.ones_like(states[0]),
				self.recurrent_dropout,
				training=training,
				count=4)

		# dropout matrices for input units
		dp_mask = self._dropout_mask
		# dropout matrices for recurrent units
		rec_dp_mask = self._recurrent_dropout_mask

		h_tm1 = states[0]  # previous memory state
		c_tm1 = states[1]  # previous carry state

		if self.implementation == 1:
			if 0 < self.dropout < 1.:
				inputs_i = inputs * dp_mask[0]
				inputs_f = inputs * dp_mask[1]
				inputs_c = inputs * dp_mask[2]
				inputs_o = inputs * dp_mask[3]
			else:
				inputs_i = inputs
				inputs_f = inputs
				inputs_c = inputs
				inputs_o = inputs
			x_i = K.dot(inputs_i, self.kernel_i)
			x_f = K.dot(inputs_f, self.kernel_f)
			x_c = K.dot(inputs_c, self.kernel_c)
			x_o = K.dot(inputs_o, self.kernel_o)
			if self.use_bias:
				x_i = K.bias_add(x_i, self.bias_i)
				x_f = K.bias_add(x_f, self.bias_f)
				x_c = K.bias_add(x_c, self.bias_c)
				x_o = K.bias_add(x_o, self.bias_o)

			if 0 < self.recurrent_dropout < 1.:
				h_tm1_i = h_tm1 * rec_dp_mask[0]
				h_tm1_f = h_tm1 * rec_dp_mask[1]
				h_tm1_c = h_tm1 * rec_dp_mask[2]
				h_tm1_o = h_tm1 * rec_dp_mask[3]
			else:
				h_tm1_i = h_tm1
				h_tm1_f = h_tm1
				h_tm1_c = h_tm1
				h_tm1_o = h_tm1
			
			z0 = x_i + K.dot(h_tm1_i,self.recurrent_kernel_i)
			z1 = x_f + K.dot(h_tm1_f,self.recurrent_kernel_f)
			z2 = x_c + K.dot(h_tm1_c,self.recurrent_kernel_c)
			z3 = x_o + K.dot(h_tm1_o, self.recurrent_kernel_o)
			
			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c1 = self.activation(z2)
			c = f * c_tm1 + i * c1
			o = self.recurrent_activation(z3)
		else:
			if 0. < self.dropout < 1.:
				inputs *= dp_mask[0]
			z = K.dot(inputs, self.kernel)
			if 0. < self.recurrent_dropout < 1.:
				h_tm1 *= rec_dp_mask[0]
			z += K.dot(h_tm1, self.recurrent_kernel)
			if self.use_bias:
				z = K.bias_add(z, self.bias)

			z0 = z[:, :self.units]
			z1 = z[:, self.units: 2 * self.units]
			z2 = z[:, 2 * self.units: 3 * self.units]
			z3 = z[:, 3 * self.units:]


			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c1 = self.activation(z2)
			c = f * c_tm1 + i * c1
			o = self.recurrent_activation(z3)

		h = o * self.activation(c)
		if 0 < self.dropout + self.recurrent_dropout:
			if training is None:
				h._uses_learning_phase = True
		#Gates
		self.gates = K.concatenate([i,f,c1,o],axis = -1)
		self.pre_gates = K.concatenate([z0,z1,z2,z3],axis = -1)
		self.h = h
		self.c = c
		return self.h, [self.h,self.c]

def lrpify_model(model):
		'''
		This function takes as input user defined keras Model, and replace all LSTM/Bi_LSTM 
		with equivalent one which have LRP_LSTMCell as core cell 
		'''
		
		cell_config_keys = ['units', 'activation', 'recurrent_activation', 'use_bias', 'unit_forget_bias', 'kernel_constraint', 'recurrent_constraint', 'bias_constraint']
		rnn_config_keys = ['return_sequences', 'return_state', 'go_backwards', 'stateful', 'unroll']
		bidirect_config_keys = ['merge_mode']
		
		
		for i,layer in enumerate(model.layers):
			if isinstance(layer,Bidirectional):
				weights = layer.get_weights()
				inp_shape = layer.input_shape
				cell_config = {key:layer.get_config()['layer']['config'][key] for key in cell_config_keys}
				rnn_config  = {key:layer.get_config()['layer']['config'][key] for key in rnn_config_keys}
				bidirect_config = {key:layer.get_config()[key] for key in bidirect_config_keys}
				
				with CustomObjectScope({'LRP_LSTMCell': LRP_LSTMCell}):
					cell = LRP_LSTMCell(**cell_config, implementation=1)
					bi_lstm = Bidirectional(RNN(cell,**rnn_config),**bidirect_config)
					bi_lstm.build(inp_shape)
					bi_lstm.call(layer.input)
					bi_lstm._inbound_nodes = layer._inbound_nodes
					bi_lstm._outbound_nodes = layer._outbound_nodes
					bi_lstm.set_weights(weights)

				model.layers[i] = bi_lstm 
			
			if isinstance(layer,LSTM):
				weights = layer.get_weights()
				inp_shape = layer.input_shape
				cell_config = {key:layer.get_config()[key] for key in cell_config_keys}
				rnn_config  = {key:layer.get_config()[key] for key in rnn_config_keys}
				with CustomObjectScope({'LRP_LSTMCell': LRP_LSTMCell}):
					cell = LRP_LSTMCell(**cell_config,implementation=1)

					lstm = RNN(cell,**rnn_config)
					lstm.build(inp_shape)
					lstm.call(layer.input)
					lstm.set_weights(weights)
					lstm._inbound_nodes = layer._inbound_nodes
					lstm._outbound_nodes = layer._outbound_nodes
				model.layers[i] = lstm
		return model