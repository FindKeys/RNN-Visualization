#from __future__ import absolute_import
import numpy as np
import argparse
import os
import imp
import re

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import  optimizers

import sys
sys.path.append(os.path.abspath("../../../../"))
from RNN_Visualization.common_utils.LRP_utils import lrpify_model
from RNN_Visualization.common_utils.LRP_utils import lrp_viz, lrp_concat,lrp_slice,lrp_lstm_batch
from RNN_Visualization.common_utils.Mask import GradientSaliencyMask
from RNN_Visualization.common_utils.Utils import visualize

from matplotlib import pyplot as plt


if __name__ == "__main__":

	args = {}

	args['data'] = '../data/in-hospital-mortality/'
	args['target_repl_coef'] = 0.0
	args['output_dir'] = '.'
	args['mode'] = 'test'
	args['timestep'] = 1
	args['network'] = 'mimic3models/keras_models/channel_wise_lstms.py'
	args['dim'] = 8
	args['depth'] = 1
	args['batch_size'] = 8
	args['dropout'] = 8
	args['load_state'] = '../best_models/ihm/r2k_channel_wise_lstms.n8.szc4.0.d0.3.dep1.bs8.ts1.0.epoch32.test0.279926446841.state'
	args['size_coef'] = 4.0
	args['normalizer_state'] = './mimic3models/in_hospital_mortality/ihm_ts0.8.input_str:previous.start_time:zero.normalizer'
	args['small_part'] = 40
	args['batch_norm'] = 0.1
	args['rec_dropout'] = 0.1
	
	target_repl = (args['target_repl_coef'] > 0.0 and args.mode == 'train')	

	# Build readers, discretizers, normalizers
	train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args['data'], 'train'),
											 listfile=os.path.join(args['data'], 'train_listfile.csv'),
											 period_length=48.0)
	
	val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args['data'], 'train'),
										   listfile=os.path.join(args['data'], 'val_listfile.csv'),
										   period_length=48.0)
	
	discretizer = Discretizer(timestep=float(args['timestep']),
							  store_masks=True,
							  impute_strategy='previous',
							  start_time='zero')	
	discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
	cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
	
	normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
	normalizer_state = args['normalizer_state']
	if normalizer_state is None:
		normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args['timestep'], args['imputation'])
		normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
	normalizer.load_params(normalizer_state)

	#args_dict = dict(args._get_kwargs())
	args_dict = {**args}
	args_dict['header'] = discretizer_header
	args_dict['task'] = 'ihm'
	args_dict['target_repl'] = target_repl
	
	# Build the model
	print("==> using model {}".format(args['network']))	
	K.clear_session()
	model_module = imp.load_source(os.path.basename(args['network']), args['network'])
	model = model_module.Network(**args_dict)
	model.load_weights(args['load_state'])
	test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args['data'], 'test'),
											listfile=os.path.join(args['data'], 'test_listfile.csv'),
											period_length=48.0)
	ret = utils.load_data(test_reader, discretizer, normalizer, args['small_part'],
						  return_names=True)
	
	data = ret["data"][0]
	labels = ret["data"][1]
	names = ret["names"]
	model = lrpify_model(model)
	model.optimizer = optimizers.Adam(10**(-3),0.9)

	name = input("Enter target time series name")
	print("Start cashing all activations")
	index = names.index(name)
	inp_data = data[[index]]
	label = labels[index]
	prediction = int(model.predict(inp_data)[0][0]>0.5)

	storage = '../data/activations/all_activations_'+str(index)+'_l'+str(label)+'p'+str(prediction)+'.p'
	try:
		with open(storage, 'rb') as handle:
			all_activations = pickle.load(handle)
	except:
		lrp_viz(inp_data, storage, model)
		with open(storage, 'rb') as handle:
			all_activations = pickle.load(handle)

	hin = all_activations['rnn_18']['final']
	w,b = model.layers[-1].get_weights()
	hout = all_activations['dense_1']
	R_out = np.zeros_like(hin)
	pre_logits = hin * w.T 
	for index, mortality_mask in enumerate(hout<0.5):
		if mortality_mask[0] == True:  #predict 0
			R_out[index] = np.where(pre_logits[index]<0,pre_logits[index],0)
		else:
			R_out[index] = np.where(pre_logits[index]>0,pre_logits[index],0)

	eps=0.01
	bias_factor=0
	hin = all_activations['concatenate_1']
	Rout_Left_last = R_out.copy()
	
	seq_len = hin.shape[1]
	hidden_dim = 32
	input_dim = hin.shape[2]
	batch_size = hin.shape[0]
	
	lstm_w = model.layers[-3].get_weights()
	layer_name = 'rnn_18'
	R_out = lrp_lstm_batch(all_activations, hin, Rout_Left_last, batch_size, seq_len, hidden_dim, input_dim, lstm_w, False, layer_name, None, eps, bias_factor, debug=True)
	
	layer_name = 'concatenate_1'
	axis = model.layers[-4].get_config()['axis']
	keys = sorted([key for key in all_activations.keys() if key.startswith('bidirectional')])
	hin_shapes = [np.concatenate([all_activations[key]['h_Left'][:,:-1],all_activations[key]['h_Right'][:,:-1][:,::-1]],axis = -1).shape for key in keys]
	R_out = lrp_concat(hin_shapes, axis, R_out, layer_name, debug=True)
	
	##Bidiractional layers LRP
	connected_to = [key for key in all_activations.keys() if key.startswith('slice')]
	R_out_ = []
	for i,(temp, prev) in enumerate(zip(keys, connected_to)):
		hin = all_activations[prev]
		seq_len = hin.shape[1]
		hidden_dim = 4
		input_dim = hin.shape[2]
		batch_size = hin.shape[0]
		lstm_w = model.layers[2+17+i].get_weights()
		layer_name = temp
 
		res = lrp_lstm_batch(all_activations, hin, R_out[i][:,-1,:4], batch_size, seq_len, hidden_dim, input_dim, lstm_w, True, layer_name,R_out[i][:,-1,4:] , eps, bias_factor, debug=False)
		for offset in range(2,seq_len+1):
			res_temp = lrp_lstm_batch(all_activations, hin[:,:-offset+1,:], R_out[i][:,-offset,:4], batch_size, seq_len-offset+1, hidden_dim, input_dim, lstm_w, True, layer_name,R_out[i][:,-offset,4:] , eps, bias_factor, debug=False)
			res += np.append(res_temp,np.zeros((batch_size,offset-1,hin.shape[-1])),axis = 1)
		if True:
			print("{} local diff: ".format(temp), R_out[i].sum()-res.sum())
		R_out_.append(res)
  
	viz = np.zeros_like(inp_data)
	for i, slice_i in enumerate(connected_to):
		indices = model.layers[2+i].get_config()['indices']
		partial_r = lrp_slice(inp_data.shape, R_out_[i], indices, slice_i, debug=False)
		viz += partial_r

	
	gd_viz = GradientSaliencyMask(model,0)
	def get_gradient_heatmap(inp,gd):
		viz = gd_viz.get_mask(inp)
		return viz

	folder = "../data/heatmaps/"
	os.mkdir(fodler)
	main_img_name = folder + name[:-4]
	main_title = name
	main_title += '   '
	main_title += "True Label: {}".format(label)
	main_title += '   '
	main_title += "Predicted Label: {}".format(int(model.predict(inp_data)[0][0]>0.5))                                
	title = main_title + '   ' + 'Description: inp'                         
	#visualize(inp[0].T, title = title, vmin=-2, vmax=2, save=True,img_name=main_img_name)
	visualize(inp_data[0].T, title = title, vmin=-2, vmax=2, save=True, cmap='RdBu',img_name=main_img_name)
  
	lrp_viz = viz[0]#get_lrp_heatmap(k)
	grad_viz = np.abs(get_gradient_heatmap(inp_data,gd_viz))
  
	vmax = np.max(lrp_viz)
	title = main_title + '   ' + 'Description: lrp'                         
	#visualize(lrp_viz.T, title, agg='sum', vmin=-vmax, vmax=vmax,save=True,img_name=main_img_name+'_'+'lrp')
	visualize(lrp_viz.T, title, agg='sum', vmin=-vmax, vmax=vmax,save=True,img_name=main_img_name+'_'+'lrp')
	
	vmax = np.max(grad_viz)
	title = main_title + '   ' + 'Description: grd'                         
	#visualize(grad_viz.T,title, agg='sum', vmin=-vmax, vmax=vmax,save=True,img_name=main_img_name+'_'+'gradient')
	visualize(grad_viz.T, title, agg='sum', vmin=-vmax, vmax=vmax,save=True,img_name=main_img_name+'_'+'gradient')









