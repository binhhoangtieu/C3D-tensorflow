#!/usr/bin/env python3
# encoding: utf-8
import cv2
import tensorflow as tf
import sys, os, h5py
import numpy as np
import tensorflow.contrib.layers as layers
import random
import  pandas as pd
from random import shuffle
from random import randint
from tqdm import  tqdm
import time
from input_data_v1 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import Counter

class C3dModel(object):
	train_size = 5041 #5041 for HMDB51, 9999 for ucf101
	test_size = 3321 #1742 
	ensemble_type = 1 #1 - average fusion, 2 - max fusion, 3 vote fusion

	def __init__(self,
			num_class = 101,
			keep_prob = 0.6,
			batch_size = 12,
			epoch=40,
			lr = 1e-4):
		self.IMG_WIDTH = 171
		self.IMG_HEIGHT = 128

		self.CROP_WIDTH = 112
		self.CROP_HEIGHT = 112
		self.graph = tf.Graph()
		self.num_class = num_class
		self.epoch = epoch
		self.CLIP_LENGTH = 16
		self.keep_prob = keep_prob
		self.batch_size = batch_size
		decay_epoch=10   #每5个epoch改变一次学习率


		# train clip: 9537*5 CLIP=5
		# test  clip: 3783*5 CLIP=5
		# train clip: 9537*3 CLIP=3
		# test  clip: 3783*3 CLIP=3

		# ucf101-testAF.list = 3374
		# ucf101-trainAF.list = 9946

		# ucf101-trainSAF.list = 10009
		# ucf101-testSAF.list = 3311
		
		# hmdb51-trainFSAF.list = 5041
		# hmdb51-testFSAF.list = 1724
		
		# KTH-trainFSAF.list: 470 CLIP=3
		# KTH-testFSAF.list: 129 CLIP=3

		# KTH-trainSAF.list = 457
		# KTH-testSAF.list = 142

		# KTH-train-rgb.list = 470
		# KTH-test-rgb.list = 129		

		self.n_step_epoch=int(self.train_size/batch_size)
		with self.graph.as_default():
			self.inputs = tf.placeholder(tf.float32, [None, self.CLIP_LENGTH, self.CROP_HEIGHT, self.CROP_WIDTH, 3])
			self.labels = tf.placeholder(tf.int64, [batch_size,])

			self.initializer = layers.xavier_initializer()
			self.global_step = tf.Variable(0, trainable = False, name = "global_step")
			self.lr = tf.train.exponential_decay(lr, self.global_step, int(decay_epoch*self.n_step_epoch), 1e-1, True)
			tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)

	def conv3d(self, inputs, shape, name,w_name,b_name):
		with self.graph.as_default():
			with tf.variable_scope('var_name') as var_scope:
				W = tf.get_variable(name = w_name, shape = shape, initializer = self.initializer, dtype = tf.float32)
				b = tf.get_variable(name = b_name, shape = shape[-1], initializer = tf.zeros_initializer(), dtype = tf.float32)
				tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
				tf.add_to_collection(tf.GraphKeys.BIASES, b)
			return tf.nn.relu(tf.nn.bias_add(tf.nn.conv3d(inputs, W, strides = [1, 1, 1, 1, 1], padding = "SAME"), b))
			# filter:
			# [filter_depth, filter_height, filter_width, in_channels,out_channels]
	def fc(self, inputs, shape, name,w_name,b_name,activation = True):
		with self.graph.as_default():
			with tf.variable_scope('var_name') as var_scope:
				W = tf.get_variable(name = w_name, shape = shape, initializer = self.initializer, dtype = tf.float32)
				b = tf.get_variable(name = b_name, shape = shape[-1], initializer = tf.zeros_initializer(), dtype = tf.float32)
				tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
				tf.add_to_collection(tf.GraphKeys.BIASES, b)

			if activation:
				return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, W), b))
			else:
				return tf.nn.bias_add(tf.matmul(inputs, W), b)

	# netstrucet is an orderdict with form {"conv": [shape, name]}
	def parseNet(self, net, netstruct, istraining = True):
		for key in netstruct:
			if key[0] == "conv":
				net = self.conv3d(net, key[2], key[1],key[3], key[4])
			elif key[0] == "fc":
				net = self.fc(net, key[2], key[1], key[3], key[4],activation = key[-1])
			elif key[0] == "maxpool":
				net = tf.nn.max_pool3d(net, ksize = key[2], strides = key[2], padding = "SAME", name = key[1])
			elif key[0] == "dropout" and istraining:
				net = tf.nn.dropout(net, key[2], name = key[1])
			elif key[0] == "reshape":
				net = tf.reshape(net, key[-1])
			elif key[0] == "softmax":
				net = tf.nn.softmax(net)
			elif key[0] == "transpose":
				net = tf.transpose(net, perm=key[-1])
		return net

	def test(self, modelpath):
		with self.graph.as_default():
			c3d_net = [
				["conv", "conv1", [3, 3, 3, 3, 64], 'wc1', 'bc1'],
				["maxpool", "pool1", [1, 1, 2, 2, 1]],
				["conv", "conv2", [3, 3, 3, 64, 128], 'wc2', 'bc2'],
				["maxpool", "pool2", [1, 2, 2, 2, 1]],
				["conv", "conv3a", [3, 3, 3, 128, 256], 'wc3a', 'bc3a'],
				["conv", "conv3b", [3, 3, 3, 256, 256], 'wc3b', 'bc3b'],
				["maxpool", "pool3", [1, 2, 2, 2, 1]],
				["conv", "conv4a", [3, 3, 3, 256, 512], 'wc4a', 'bc4a'],
				["conv", "conv4b", [3, 3, 3, 512, 512], 'wc4b', 'bc4b'],
				["maxpool", "pool4", [1, 2, 2, 2, 1]],
				["conv", "conv5a", [3, 3, 3, 512, 512], 'wc5a', 'bc5a'],
				["conv", "conv5b", [3, 3, 3, 512, 512], 'wc5b', 'bc5b'],
				["maxpool", "pool5", [1, 2, 2, 2, 1]],
				["transpose", [0, 1, 4, 2, 3]],  #only use it if you restore the sports1m_finetuning_ucf101.model, otherwise uncomment it,(e.g use conv3d_deepnetA_sport1m_iter_1900000_TF.model)
				["reshape", [-1, 8192]],
				["fc", "fc1", [8192, 4096], 'wd1', 'bd1', True],
				["dropout", "dropout1", self.keep_prob],
				["fc", "fc2", [4096, 4096],'wd2','bd2', True],
				["dropout", "dropout2", self.keep_prob],
				["fc", "fc3", [4096, self.num_class],'wout','bout',False],
			]

			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			config.gpu_options.per_process_gpu_memory_fraction = 0.9

			with tf.Session(config=config, graph=self.graph) as sess:
				logits = self.parseNet(self.inputs, c3d_net)
				softmax_logits = tf.nn.softmax(logits)

				int_label = self.labels 

				task_loss = tf.reduce_sum(
					tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=int_label))

				acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=-1), int_label), tf.float32))
				right_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=1), int_label), tf.int32))
				ensemble_logist = softmax_logits
				reg_loss = layers.apply_regularization(layers.l2_regularizer(5e-4),
													   tf.get_collection(tf.GraphKeys.WEIGHTS))
				total_loss = task_loss + reg_loss
				
				train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(
					total_loss, global_step=self.global_step)
	
				total_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
				print('total_para:', total_para) 

				init = tf.global_variables_initializer()

				sess.run(init)
				saver = tf.train.Saver(tf.trainable_variables())
				
				# ========================================================================================
				#Recode after lost all code - awful day 21/5/2018

				# test_list=["./test1.list",'./test1.list',"./test1.list"]
				# test_list=["./kth_rgb_test.list",'./kth_fsaf_test2.list',"./kth_of_test2.list"]
				# network_models = ['c3d_kth_rgb','c3d_kth_fsaf','c3d_kth_of']

				test_list=["./hmdb51_rgb_test.list","./hmdb51_fsaf_test.list",'./hmdb51_of_test2.list']
				network_models = ['c3d_hmdb51_rgb','c3d_hmdb51_fsaf','c3d_hmdb51_of']

				# test_list=["./ucf101_rgb_test.list","./ucf101_saf_test2.list",'./ucf101_of_test2.list']
				# network_models = ['c3d_ucf_rgb','c3d_ucf_saf','c3d_ucf_of']

				# lines = open(test_list[0],'r')
				# # lines = list(lines)
				# lines = list(line for line in lines if line)
				# number_of_line = len(lines)
				# self.test_size = number_of_line
				list_accuracy = []
				pred_labels = []
				true_labels = []
				num_networks = len(network_models)
				# ======================================================================================
				for m in range(num_networks):
					softmax_one_networks = []
					saver.restore(sess, modelpath + network_models[m])
					print("Model {:2d} loading finished!" .format(m))
					step = 0
					print_freq = 2
					next_start_pos = 0
					lines = open(test_list[m],'r')
					# lines = list(lines)
					lines = list(line for line in lines if line)
					number_of_line = len(lines)
					self.test_size = number_of_line
					# print(number_of_line)
					for one_epoch in range(1):
						epostarttime = time.time()
						starttime = time.time()
						total_v = 0.0
						test_correct_num = 0
						
						for i in tqdm(range(int(self.test_size / self.batch_size))):
							step += 1
							total_v += self.batch_size
							
							train_batch, label_batch, next_start_pos, _,_ = read_clip_and_label(
								filename=test_list[m],
								batch_size=self.batch_size,
								start_pos=next_start_pos,
								num_frames_per_clip=self.CLIP_LENGTH,
								height=self.IMG_HEIGHT,
								width=self.IMG_WIDTH,
								shuffle=False)

							assert len(train_batch)==self.batch_size
							train_batch = train_aug(train_batch, is_train=False, Crop_heith=self.CROP_HEIGHT,
													Crop_width=self.CROP_WIDTH,norm=True)
							val_feed = {self.inputs: train_batch, self.labels: label_batch}
							test_correct_num += sess.run(right_count, val_feed)
							
							#add 22/5
							softmax = sess.run(ensemble_logist, val_feed)
							if m == 0: #get for first network only
								true_labels.extend(label_batch)

							softmax_one_networks.extend(softmax)

							print('test acc:', test_correct_num / total_v, 'test_correct_num:', test_correct_num,
								  'total_v:', total_v)
					list_accuracy.append(test_correct_num / total_v)
					pred_labels.append(softmax_one_networks)

				print(list_accuracy)
				print(np.shape(true_labels),np.shape(pred_labels))
				# pred_labels shape = (num_networks, num_label,num_class)
				# print(true_labels)
				
				#ensemble:
				number_of_test = len(true_labels)
				if self.ensemble_type == 1: #average fusion
					ensemble_pred_labels = np.mean(pred_labels, axis = 0)
					ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
					
				elif self.ensemble_type == 2: # max average
					ensemble_pred_labels = np.amax(pred_labels, axis = 0)
					ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
				else: #vote fusion
					#Compare networks
					
					vote_softmax = np.zeros(number_of_test,dtype = int)
					print(number_of_test,np.shape(pred_labels))
					for i in range(number_of_test):
						argmax_networks = []
						for m in range(num_networks):
							argmax_networks.append(np.argmax(pred_labels[m][i],axis=0))
						# compare each network to choose 
						counter = Counter(argmax_networks)
						best_net = [(k, v) for k, v in counter.items() if v == max(counter.values())]
						if len(best_net) > 1: #there are many network with predict the same label
							vote_softmax[i] = np.argmax(np.amax(pred_labels, axis = 0), axis=1)[i]
							# print(best_net,i,vote_softmax[i],true_labels[i])
						else:
							vote_softmax[i] = best_net[0][0]
					ensemble_cls_pred = vote_softmax
					
				ensemble_correct = (ensemble_cls_pred == true_labels)
				print('ensemble accuracy:', np.sum(ensemble_correct/number_of_test))


if __name__ == "__main__":
	c3dnet = C3dModel()
	c3dnet.test(modelpath="./models/hmdb51-all/")