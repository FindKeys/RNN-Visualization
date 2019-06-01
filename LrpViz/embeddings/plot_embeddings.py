import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

def plot_with_tensorboard():
	os.chdir('..')
	PATH = os.getcwd()
	vocab_path = PATH + '/embeddings/vocab.dms'
	embd_path = PATH + '/embeddings/embeddings.npy'
	LOG_DIR = PATH + '/tensorboard/log'
	metadata = os.path.join(LOG_DIR, 'metadata.tsv')

	with open(vocab_path,'rb') as f_voc:
		voc  = pickle.load(f_voc)
		f_voc.close()

	E = np.load(embd_path, mmap_mode='r')
	N = E.shape[0]
	assert N == len(voc)
	
	embeddings = tf.Variable(E, name='words')
	print('write metadata')

	with open(metadata, 'wt') as metadata_file:
		metadata_file.write('{}\n'.format('Index\tWord'))
		for i,word in tqdm(zip(np.arange(N),voc)):
			c = str(i) +'\t'+ str(word)
			metadata_file.write('{}\n'.format(c))


	print('Project embeddings')
	with tf.Session() as sess:
		saver = tf.train.Saver([embeddings])

		sess.run(embeddings.initializer)
		saver.save(sess, os.path.join(LOG_DIR, 'word_embeddings.ckpt'))

		config = projector.ProjectorConfig()
		# One can add multiple embeddings.
		embedding = config.embeddings.add()
		embedding.tensor_name = embeddings.name
		# Link this tensor to its metadata file (e.g. labels).
		embedding.metadata_path = metadata
		# Saves a config file that TensorBoard will read during startup.
		projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

if __name__ == '__main__':
	plot_with_tensorboard()
	