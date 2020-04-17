"""Training and testing the Deep Listwise Context Model.

See the following paper for more information.
	
	* Qingyao Ai, Keping Bi, Jiafeng Guo, W. Bruce Croft. 2018. Learning a Deep Listwise Context Model for Ranking Refinement. In Proceedings of SIGIR '18
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin
from .BasicRankingModel import BasicRankingModel
import copy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


# TODO(ebrevdo): Remove once _linear is fully deprecated.
#linear = rnn_cell_impl._linear  # pylint: disable=protected-access
linear = core_rnn_cell._linear 
class RankLSTM(BasicRankingModel):
# 	def __init__(self,data_set, exp_settings, forward_only):
# 		"""Create the model.

# 		"""
	def __init__(self,hparams):
		self.hparams = tf.contrib.training.HParams(
			learning_rate=0.1, 				# Learning rate.
			learning_rate_decay_factor=0.8, # Learning rate decays by this much.
			max_gradient_norm=5.0,			# Clip gradients to this norm.
			reverse_input=True,				# Set to True for reverse input sequences.
			num_layers=1,					# Number of layers in the model.
			num_heads=3,					# Number of heads in the attention strategy.
			loss_func='attrank',			# Select Loss function
			l2_loss=0.0,					# Set strength for L2 regularization.
			att_strategy='add',				# Select Attention strategy
			use_residua=False,				# Set to True for using the initial scores to compute residua.
			use_lstm=False,					# Set to True for using LSTM cells instead of GRU cells.
			softRank_theta=0.1,				# Set Gaussian distribution theta for softRank.
		)
		print("building DLCM")     
		self.hparams.parse(hparams)
		self.start_index = 0
		self.count = 1
# 		self.rank_list_size = exp_settings['max_candidate_num']
# 		self.embed_size = data_set.feature_size if data_set.feature_size > 0 else 0
		self.expand_embed_size =50
		self.feed_previous=False
# 		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * self.hparams.learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		
		output_projection = None

		self.output_projection=output_projection
		
# 		dtype=dtypes.float32
# 		cel=self.cell
		scope=None
		# If we use sampled softmax, we need an output projection.
		output_projection = None
		
		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
# 		self.embeddings = tf.placeholder(tf.float32, shape=[None, embed_size], name="embeddings")
		self.target_labels = []
		self.target_weights = []
		self.target_initial_score = []
		with variable_scope.variable_scope("embedding_rnn_seq2seq",reuse=tf.AUTO_REUSE):
			self.batch_embedding=tf.keras.layers.BatchNormalization(name="embedding_norm")
			self.layer_norm_hidden=tf.keras.layers.LayerNormalization(name="layer_norm_state")
			self.layer_norm_final=tf.keras.layers.LayerNormalization(name="layer_norm_final")
# 		for i in xrange(self.rank_list_size):
# 			self.encoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
# 											name="encoder{0}".format(i)))
# 		for i in xrange(self.rank_list_size):
# 			self.target_labels.append(tf.placeholder(tf.int64, shape=[None],
# 										name="targets{0}".format(i)))
# 			self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
# 											name="weight{0}".format(i)))
# 			self.target_initial_score.append(tf.placeholder(tf.float32, shape=[None],
# 											name="initial_score{0}".format(i)))
		# Create the internal multi-layer cell for our RNN.


# 		self.batch_index_bias = tf.placeholder(tf.int64, shape=[None])
# 		self.batch_expansion_mat = tf.placeholder(tf.float32, shape=[None,1])
# 		self.batch_diag = tf.placeholder(tf.float32, shape=[None,self.rank_list_size,self.rank_list_size])
# 		self.GO_embed = tf.get_variable("GO_embed", [1,self.embed_size + expand_embed_size],dtype=tf.float32)
# 		self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)
        
# 		self.outputs, self.state= self.embedding_rnn_seq2seq(self.encoder_inputs, self.embeddings,
# 													 cell,	output_projection, forward_only or feed_previous)

	def _extract_argmax_and_embed(self,embedding, output_projection=None,
								 update_embedding=True):
		"""Get a loop_function that extracts the previous symbol and embeds it.

		Args:
		embedding: embedding tensor for symbols.
		output_projection: None or a pair (W, B). If provided, each fed previous
			output will first be multiplied by W and added B.
		update_embedding: Boolean; if False, the gradients will not propagate
			through the embeddings.

		Returns:
		A loop function.
		"""

		def loop_function(prev, _):
			if output_projection is not None:
				prev = nn_ops.xw_plus_b(
					prev, output_projection[0], output_projection[1])
			prev_symbol = math_ops.argmax(prev, 1) + tf.to_int64(self.batch_index_bias)
			# Note that gradients will not propagate through the second parameter of
			# embedding_lookup.
			emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
			if not update_embedding:
				emb_prev = tf.stop_gradient(emb_prev)
			return emb_prev

		return loop_function


	def rnn_decoder(self,encode_embed, attention_states, initial_state, cell, 
					 num_heads=1, loop_function=None, dtype=dtypes.float32, scope=None,
					 initial_state_attention=False):
		"""RNN decoder for the sequence-to-sequence model.

		"""
		with variable_scope.variable_scope(scope or "rnn_decoder"):
			batch_size = tf.shape(encode_embed[0])[0]# Needed for reshaping.
			attn_length = attention_states.get_shape()[1].value #number of output vector in sequence
			attn_size = attention_states.get_shape()[2].value #the dimension size of each output vector
			state_size = initial_state.get_shape()[1].value #the dimension size of state vector
			print(batch_size,attn_length,attn_size,state_size,"batch_size,attn_lengt,attn_size,state_size")
			# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
			print(attention_states.get_shape(),"attention_states.get_shape()")
			hidden = tf.reshape(
				attention_states, [-1, attn_length, 1, attn_size])
			hidden_features = []
			hidden_features2 = []
			v = []
			u = []
			linear_w = []
			linear_b = []
			abstract_w = []
			abstract_b = []
			abstract_layers = [int((attn_size + state_size)/(2 + 2*i)) for i in xrange(2)] + [1]
			attention_vec_size = attn_size# Size of query vectors for attention.
			head_weights = []
			for a in xrange(num_heads):
				k = variable_scope.get_variable("AttnW_%d" % a,
												[1, 1, attn_size, attention_vec_size]) 
				hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))#[B,T,1,attn_vec_size]
				k2 = variable_scope.get_variable("AttnW2_%d" % a,
												[1, 1, attn_size, attention_vec_size])
				hidden_features2.append(nn_ops.conv2d(hidden, k2, [1, 1, 1, 1], "SAME"))
				v.append(variable_scope.get_variable("AttnV_%d" % a,
													 [attention_vec_size]))
				u.append(variable_scope.get_variable("AttnU_%d" % a,
													 [attention_vec_size]))
				head_weights.append(variable_scope.get_variable("head_weight_%d" % a,[1]))
				current_layer_size = attn_size + state_size
				linear_w.append(variable_scope.get_variable("linearW_%d" % a,
													 [1,1,current_layer_size, 1]))
				linear_b.append(variable_scope.get_variable("linearB_%d" % a,
													 [1]))
				abstract_w.append([])
				abstract_b.append([])
				for i in xrange(len(abstract_layers)):
					layer_size = abstract_layers[i]
					abstract_w[a].append(variable_scope.get_variable("Att_%d_layerW_%d" % (a,i),
													 [1,1,current_layer_size, layer_size]))
					abstract_b[a].append(variable_scope.get_variable("Att_%d_layerB_%d" % (a,i),
													 [layer_size]))
					current_layer_size = layer_size
				

			def attention(query):
				"""Put attention masks on hidden using hidden_features and query."""
				ds = []# Results of attention reads will be stored here.
				aw = []# Attention weights will be stored here
				tiled_query = tf.tile(tf.reshape(query, [-1, 1, 1, state_size]),[1,attn_length,1, 1])
				print(hidden.get_shape(),"hidden.get_shape()")
				print(tiled_query.get_shape(),"tiled_query.get_shape()")
				concat_input = tf.concat(axis=3, values=[hidden, tiled_query])
				#concat_input = tf.concat(3, [hidden, hidden])
				for a in xrange(num_heads):
					with variable_scope.variable_scope("Attention_%d" % a):
						s = None
						if self.hparams.att_strategy == 'multi':
							print('Attention: multiply')
							y = linear(query, attention_vec_size, True)
							y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
							#s = math_ops.reduce_sum(
							#	u[a] * math_ops.tanh(y * hidden_features[a]), [2, 3])
							s = math_ops.reduce_sum(
								hidden * math_ops.tanh(y), [2, 3])
								#hidden_features[a] * math_ops.tanh(y), [2, 3])

						elif self.hparams.att_strategy == 'multi_add':
							print('Attention: multiply_add')
							y = linear(query, attention_vec_size, True, scope='y')
							y2 = linear(query, attention_vec_size, True , scope='y2')
							y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
							y2 = tf.reshape(y2, [-1, 1, 1, attention_vec_size])
							#s = math_ops.reduce_sum(
							#	u[a] * math_ops.tanh(y * hidden_features[a]), [2, 3])
							s = math_ops.reduce_sum(
								hidden * math_ops.tanh(y2), [2, 3])
							s = s + math_ops.reduce_sum(
								v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

						elif self.hparams.att_strategy == 'NTN':
							print('Attention: NTN')
							y = linear(query, attn_size, False)
							y = tf.tile(tf.reshape(y, [-1, 1, 1, attn_size]),[1,attn_length,1,1])
							s = math_ops.reduce_sum(hidden * y, [2,3]) #bilnear
							s = s + math_ops.reduce_sum(nn_ops.conv2d(concat_input, linear_w[a], [1, 1, 1, 1], "SAME"), [2,3]) #linear
							s = s + linear_b[a] #bias
							#print(s.get_shape())
							#s = tf.tanh(s) #non linear

						elif self.hparams.att_strategy == 'elu':
							print('Attention: elu')

							cur_input = concat_input
							#for i in xrange(len(abstract_layers)):
							#	cur_input = tf.contrib.layers.fully_connected(cur_input, abstract_layers[i], activation_fn=tf.nn.elu)
							for i in xrange(len(abstract_layers)):
								cur_input = nn_ops.conv2d(cur_input, abstract_w[a][i], [1, 1, 1, 1], "SAME")
								cur_input = cur_input + abstract_b[a][i]
								cur_input = tf.nn.elu(cur_input)
							s = math_ops.reduce_sum(cur_input,[2,3])

						else:
							print('Attention: add')
							y = linear(query, attention_vec_size, True)
							y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
							s = math_ops.reduce_sum(
								v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

						att = s * head_weights[a]#nn_ops.softmax(s)
						aw.append(att)
						# Now calculate the attention-weighted vector d.
						d = math_ops.reduce_sum(
							tf.reshape(att, [-1, attn_length, 1, 1]) * hidden,
								[1, 2])
						ds.append(tf.reshape(d, [-1, attn_size]))
				return aw, ds


			state = initial_state
			outputs = []
			prev = None
			batch_attn_size = tf.stack([batch_size, attn_size])
			batch_attw_size = tf.stack([batch_size, attn_length])
			attns = [tf.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]
			attw = [1.0/attn_length * tf.ones(batch_attw_size, dtype=dtype) for _ in xrange(num_heads)]
			for a in attns:# Ensure the second shape of attention vectors is set.
				a.set_shape([None, attn_size])

			# Directly use previous state
			attw, attns = attention(initial_state)
			aw = math_ops.reduce_sum(attw,0)
			output = tf.scalar_mul(1.0/float(num_heads), aw)
			output = output - tf.reduce_min(output,1,keep_dims=True)
			outputs.append(output)

		return outputs, state


	def embedding_rnn_decoder(self,initial_state, cell,
							attention_states, encode_embed, num_heads=1,
							 output_projection=None,
							 feed_previous=False,
							 update_embedding_for_previous=True, scope=None):
		"""RNN decoder with embedding and a pure-decoding option.

		"""
		if output_projection is not None:
			proj_weights = ops.convert_to_tensor(output_projection[0],
												 dtype=dtypes.float32)
			proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
			proj_biases = ops.convert_to_tensor(
				output_projection[1], dtype=dtypes.float32)
			proj_biases.get_shape().assert_is_compatible_with([num_symbols])

		with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
			loop_function = self._extract_argmax_and_embed(
				encode_embed, output_projection,
				update_embedding_for_previous) if feed_previous else None
			# emb_inp = (
			#	embedding_ops.embedding_lookup(embeddings, i) for i in decoder_inputs)
			#emb_inp = decoder_embed
			return self.rnn_decoder(encode_embed, attention_states, initial_state, cell,
								num_heads=num_heads, loop_function=loop_function)
	def build_with_random_noise(self, input_list, noise_rate, is_training=False):
			return self.build(input_list, noise_rate, is_training)
	def build(self, encoder_embed,is_training=False):
		"""Embedding RNN sequence-to-sequence model.

		
		"""
		print(encoder_embed[0].get_shape(),"encoder_embed.get_shape()")
		feed_previous=self.feed_previous
		embed_size = encoder_embed[0].get_shape()[-1].value
		dtype=dtypes.float32
		output_projection=self.output_projection
		list_size=len(encoder_embed)
		with variable_scope.variable_scope("cell",reuse=tf.AUTO_REUSE):
			single_cell = tf.contrib.rnn.GRUCell(embed_size + self.expand_embed_size)
			double_cell = tf.contrib.rnn.GRUCell((embed_size +self.expand_embed_size) * 2)
			if self.hparams.use_lstm:
				single_cell = tf.contrib.rnn.BasicLSTMCell((embed_size + self.expand_embed_size))
				double_cell = tf.contrib.rnn.BasicLSTMCell((embed_size + self.expand_embed_size) * 2)
			cell = single_cell
			self.double_cell = double_cell
			self.cell=cell
			self.output_projection=output_projection

			if self.hparams.num_layers > 1:
				cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.hparams.num_layers)
				self.double_cell = tf.contrib.rnn.MultiRNNCell([double_cell] * self.hparams.num_layers)            
		with variable_scope.variable_scope(tf.get_variable_scope() or "embedding_rnn_seq2seq",reuse=tf.AUTO_REUSE):
			
			
			def abstract(input_data, index):
				reuse = None if index < 1 else True
				print(reuse,"reuse or not",tf.AUTO_REUSE,"tf.AUTO_REUSE")
				with variable_scope.variable_scope(variable_scope.get_variable_scope(),
												 reuse=tf.AUTO_REUSE):
					output_data = input_data
					output_sizes = [int((embed_size + self.expand_embed_size)/2), self.expand_embed_size]
					current_size = embed_size
					for i in xrange(2):
						expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]])
						expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
						output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
						output_data = tf.nn.elu(output_data)
						current_size = output_sizes[i]
					return output_data
			for i in xrange(list_size):
# 				encoder_embed.append(embedding_ops.embedding_lookup(embeddings, encoder_inputs[i]))
				#expand encoder size
				encoder_embed[i]=self.batch_embedding(encoder_embed[i])
				if self.expand_embed_size > 0:
					encoder_embed[i] =  tf.concat(axis=1, values=[encoder_embed[i], abstract(encoder_embed[i], i)])
# 			print(len(encoder_embed),encoder_embed[0].get_shape(),"encoder_embed.get_shape()")
			enc_cell = copy.deepcopy(cell)
			ind=list(range(0,list_size))
			random.shuffle(ind)
# 			encoder_embed_input=[encoder_embed[i] for i in ind ]
			encoder_embed_input=encoder_embed[::-1]
			encoder_outputs_random, encoder_state = tf.nn.static_rnn(enc_cell, encoder_embed_input, dtype=dtype)
# 			encoder_outputs=[None]*list_size
# 			for i in range(list_size):
# 				encoder_outputs[ind[i]]=encoder_outputs_random[i]     
			encoder_outputs=encoder_outputs_random[::-1]
# 			print(len(encoder_outputs),"encoder_outputs.get_shape()",\
#                   encoder_state.get_shape(),"encoder_state.get_shape(),")
# 			encoder_outputs=encoder_outputs
# 			top_states = [tf.reshape(self.layer_norm_hidden(e), [-1, 1, cell.output_size])
# 						for e in encoder_outputs]
# 			encoder_state=self.layer_norm_final(encoder_state)
			top_states = [tf.reshape(e, [-1, 1, cell.output_size])
						for e in encoder_outputs]
			encoder_state=encoder_state
						#for e in encoder_embed]
# 			print(len(top_states),top_states[0].get_shape(),"top_states[0].get_shape()")
			attention_states = tf.concat(axis=1, values=top_states)
# 			print(attention_states.get_shape(),"attention_states.get_shape()")
			'''
			# Concat encoder inputs with encoder outputs to form attention vector
			input_states = [tf.reshape(e, [-1, 1, cell.output_size])
						for e in encoder_embed]
			input_att_states = tf.concat(1, input_states)
			attention_states = tf.concat(2, [input_att_states, attention_states])
			'''
			'''
			# Concat hidden states with encoder outputs to form attention vector
			hidden_states = [tf.reshape(e, [-1, 1, cell.output_size])
						for e in encoder_states]
			hidden_att_states = tf.concat(1, hidden_states)
			attention_states = tf.concat(2, [hidden_att_states, attention_states])
			'''

			# Decoder.
			#GO = tf.matmul(batch_expansion_mat, self.GO_embed)
			#decoder_embed_patched = [GO] + decoder_embed


			if isinstance(feed_previous, bool):
				outputs, state=self.embedding_rnn_decoder(
					encoder_state, cell, attention_states,encoder_embed,
					num_heads=self.hparams.num_heads, output_projection=output_projection,
					feed_previous=feed_previous)
				print(outputs[0].get_shape(),"outputs[0].get_shape()")
				return outputs[0]

			# If feed_previous is a Tensor, we construct 2 graphs and use cond.
			def decoder(feed_previous_bool):
				reuse = None if feed_previous_bool else True
				with variable_scope.variable_scope(variable_scope.get_variable_scope(),
												 reuse=reuse):
					outputs, state = self.embedding_rnn_decoder(
						encoder_state, cell, attention_states,encoder_embed,
						num_heads = self.hparams.num_heads, output_projection=output_projection,
						feed_previous=feed_previous_bool,
						update_embedding_for_previous=False)
					return outputs + [state]

			outputs_and_state = control_flow_ops.cond(feed_previous,
													lambda: decoder(True),
													lambda: decoder(False))
			print(outputs[0].get_shape(),"outputs[0].get_shape()")
			return outputs_and_state[0]

	def attrank_loss(self, output, target_indexs, target_rels, name=None):
		loss = 600
		with ops.name_scope(name, "attrank_loss",[output] + target_indexs + target_rels):
			target = tf.transpose(ops.convert_to_tensor(target_rels))
			#target = tf.nn.softmax(target)
			#target = target / tf.reduce_sum(target,1,keep_dims=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)
		batch_size = tf.shape(target_rels[0])[0]
		return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

	def pairwise_loss(self, output, target_indexs, target_rels, name=None):
		loss = 0
		batch_size = tf.shape(target_rels[0])[0]
		with ops.name_scope(name, "pairwise_loss",[output] + target_indexs + target_rels):
			for i in xrange(batch_size):
				for j1 in xrange(self.rank_list_size):
					for j2 in xrange(self.rank_list_size):
						if output[i][j1] > output[i][j2] and target_rels[i][j1] < target_rels[i][j2]:
							loss += target_rels[i][j2] - target_rels[i][j1]
		return loss

		
		return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

	def listMLE(self, output, target_indexs, target_rels, name=None):
		loss = None
		with ops.name_scope(name, "listMLE",[output] + target_indexs + target_rels):
			output = tf.nn.l2_normalize(output, 1)
			loss = -1.0 * math_ops.reduce_sum(output,1)
			print(loss.get_shape())
			exp_output = tf.exp(output)
			exp_output_table = tf.reshape(exp_output,[-1])
			print(exp_output.get_shape())
			print(exp_output_table.get_shape())
			sum_exp_output = math_ops.reduce_sum(exp_output,1)
			loss = tf.add(loss, tf.log(sum_exp_output))
			#compute MLE
			for i in xrange(self.rank_list_size-1):
				idx = target_indexs[i] + tf.to_int64(self.batch_index_bias)
				y_i = embedding_ops.embedding_lookup(exp_output_table, idx)
				#y_i = tf.gather_nd(exp_output, idx)
				sum_exp_output = tf.subtract(sum_exp_output, y_i)
				loss = tf.add(loss, tf.log(sum_exp_output))
		batch_size = tf.shape(target_rels[0])[0]
		return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)

	def softRank(self, output, target_indexs, target_rels, name=None):
		loss = None
		batch_size = tf.shape(target_rels[0])[0]
		theta = 0.1
		with ops.name_scope(name, "softRank",[output] + target_indexs + target_rels):
			output = tf.nn.l2_normalize(output, 1)
			#compute pi_i_j
			tmp = tf.concat(axis=1, values=[self.batch_expansion_mat for _ in xrange(self.rank_list_size)])
			tmp_expand = tf.expand_dims(tmp, -2)
			output_expand = tf.expand_dims(output, -2)
			dif = tf.subtract(tf.matmul(tf.matrix_transpose(output_expand), tmp_expand),
							tf.matmul(tf.matrix_transpose(tmp_expand), output_expand))
			#unpacked_pi = self.integral_Guaussian(dif, theta)
			unpacked_pi = tf.add(self.integral_Guaussian(dif, self.hparams.softRank_theta), self.batch_diag) #make diag equal to 1.0
			#may need to unpack pi: pi_i_j is the probability that i is bigger than j
			pi = tf.unstack(unpacked_pi, None, 1)
			for i in xrange(self.rank_list_size):
				pi[i] = tf.unstack(pi[i], None, 1)
			#compute rank distribution p_j_r
			one_zeros = tf.matmul(self.batch_expansion_mat, 
						tf.constant([1.0]+[0.0 for r in xrange(self.rank_list_size-1)], tf.float32, [1,self.rank_list_size]))
			#initial_value = tf.unpack(one_zeros, None, 1)
			pr = [one_zeros for _ in xrange(self.rank_list_size)] #[i][r][None]
			#debug_pr_1 = [one_zeros for _ in xrange(self.rank_list_size)] #[i][r][None]
			for i in xrange(self.rank_list_size):
				for j in xrange(self.rank_list_size):
					#if i != j: #insert doc j
					pr_1 = tf.pad(tf.stack(tf.unstack(pr[i], None, 1)[:-1],1), [[0,0],[1,0]], mode='CONSTANT')
					#debug_pr_1[i] = pr_1
						#pr_1 = tf.concat(1, [self.batch_expansion_mat*0.0, tf.unpack(pr[i], None, 1)[:-1]])
					factor = tf.tile(tf.expand_dims(pi[i][j], -1),[1,self.rank_list_size])
						#print(factor.get_shape())
					pr[i] = tf.add(tf.multiply(pr[i], factor),
									tf.multiply(pr_1, 1.0 - factor))
						#for r in reversed(xrange(self.rank_list_size)):
							#if r < 1:
							#	pr[i][r] = tf.mul(pr[i][r], pi[i][j])
							#else:
							#	pr[i][r] = tf.add(tf.mul(pr[i][r], pi[i][j]),
							#			tf.mul(pr[i][r-1], 1.0 - pi[i][j]))

			#compute expected NDCG
			#compute Gmax
			Dr = tf.matmul(self.batch_expansion_mat, 
					tf.constant([1.0/math.log(2.0+r) for r in xrange(self.rank_list_size)], tf.float32, [1,self.rank_list_size]))
			gmaxs = []
			for i in xrange(self.rank_list_size):
				idx = target_indexs[i] + tf.to_int64(self.batch_index_bias)
				g = embedding_ops.embedding_lookup(target_rels, idx)
				gmaxs.append(g)
			_gmax = tf.exp(tf.stack(gmaxs, 1)) * (1.0 / math.log(2))
			Gmax = tf.reduce_sum(tf.multiply(Dr, _gmax), 1)
			#compute E(Dr)
			Edrs = []
			for i in xrange(self.rank_list_size):
				edr = tf.multiply(Dr, pr[i])
				Edrs.append(tf.reduce_sum(edr,1))
			#compute g(j)
			g = tf.exp(tf.stack(target_rels, 1)) * (1.0 / math.log(2))
			dcg = tf.multiply(g, tf.stack(Edrs, 1))
			Edcg = tf.reduce_sum(dcg, 1)
			Ndcg = tf.div(Edcg, Gmax)
			#compute loss
			loss = (Ndcg * -1.0 + 1) * 10
		return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32)#, pi, pr, Ndcg]

	def integral_Guaussian(self, mu, theta):
		a = -4.0/math.sqrt(2.0*math.pi)/theta
		exp_mu = tf.exp(a * mu)
		ig = tf.div(exp_mu, exp_mu + 1) * -1.0 + 1
		return ig
	def clip_by_each_value(self, t_list, clip_max_value = None, clip_min_value = None, name=None):
		if (not isinstance(t_list, collections.Sequence)
			or isinstance(t_list, six.string_types)):
			raise TypeError("t_list should be a sequence")
		t_list = list(t_list)

		with ops.name_scope(name, "clip_by_each_value",t_list + [clip_norm]) as name:
			values = [
					ops.convert_to_tensor(
							t.values if isinstance(t, ops.IndexedSlices) else t,
							name="t_%d" % i)
					if t is not None else t
					for i, t in enumerate(t_list)]

			values_clipped = []
			for i, v in enumerate(values):
				if v is None:
					values_clipped.append(None)
				else:
					t = None
					if clip_value_max != None:
						t = math_ops.minimum(v, clip_value_max)
					if clip_value_min != None:
						t = math_ops.maximum(t, clip_value_min, name=name)
					with ops.colocate_with(t):
						values_clipped.append(
								tf.identity(t, name="%s_%d" % (name, i)))

			list_clipped = [
					ops.IndexedSlices(c_v, t.indices, t.dense_shape)
					if isinstance(t, ops.IndexedSlices)
					else c_v
					for (c_v, t) in zip(values_clipped, t_list)]

		return list_clipped


