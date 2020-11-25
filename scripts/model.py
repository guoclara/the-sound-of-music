import tensorflow as tf
import numpy as np

# NOTE: spectrograms should be reshaped to [-1, 128, 128, 1] and of dtype tf.float32

class BasicCNN(tf.keras.Model):
	def __init__(self):
		"""
		Just a basic CNN architecture.
		"""
		super(BasicCNN, self).__init__()
      
		# Defining hyperparameters and optimizer
		self.batch_sz = 20
		self.learn_rate = 0.001
		self.dropout_rate = 0.4
		self.hlayer_sz = 1024
		self.optimizer = tf.keras.optimizers.Adam(self.learn_rate)

		# NOTE: use_bias for Conv2D is 'true' on default
		# Define model layers for convolution
		self.conv1 = tf.keras.layers.Conv2D(32, (3,3), (1,1), 'same', activation='relu')
		self.pool1 = tf.keras.layers.MaxPool2D((2,2), strides=2)
		self.conv2 = tf.keras.layers.Conv2D(64, (3,3), (1,1), 'same', activation='relu')
		self.pool2 = tf.keras.layers.MaxPool2D((2,2), strides=2)

		# Define model layers for feed forward
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(self.hlayer_sz, activation='relu')
		self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
		self.dense2 = tf.keras.layers.Dense(10, activation=None)
  
	def call(self, inputs):
		"""
		Run forward pass on input batch of spectrograms
		:param inputs: batch of images, shape [batch_sz, 128, 128, 1]
		:return: logits, with shape [batch_sz, 10]
		"""
		# NOTE: there is only one channel for input images
		conv1_out = self.conv1(inputs)
		conv1_pool = self.pool1(conv1_out)
		conv2_out = self.conv2(conv1_pool)
		# TODO: add mask here (mask feature map with template for each filter)
		conv2_pool = self.pool2(conv2_out)
  
		# Flatten tensor to pass through linear layer(s)
		flattened = self.flatten(conv2_pool)
		dense_out = self.dense1(flattened)
		dropout = self.dropout(dense_out)
		logits = self.dense2(dropout) # Shape [batch_sz, 10]

		# NOTE: compute softmax on logits (and find classes) in loss func
		return logits
    

	def loss_func(self, logits, labels):
		"""
		Calculates loss for basic CNN architecture
		:param logits: raw predictions of shape [batch_sz, 10] 
		:param labels: correct labels for given batch [batch_sz]
		"""
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
		
		return tf.reduce_mean(loss)

	def accuracy_func(self, logits, labels):
		"""
		Calculates model's prediction accuracy
		:param logits: raw predictions of shape [batch_sz, 10]
		:param labels: correct labels for given batch [batch_sz]
		"""
		num_correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
		
		return tf.reduce_mean(tf.cast(num_correct_predictions, tf.float32))

	# feature map = output of filter
	def interpretable_loss(self, feature_map, labels, z):
		pass
		# is_negative = true
		# for all mu possible
			# if is_negative:
				# set p(mu) = 1-alpha
				# get negative template
				# is_negative = false
			# else
				# set p(mu) = alpha/n^2
				# get template from argmax
			# for map in feature_map
				# p(x|mu) = 1/z * exp(tr(map * template))
				# p(x) = 0
				# for all mu
					# p(x) += p(mu) * p(x|mu) = p(mu) * [1/z * exp(tr(map * template))]
				# inner_sum += p(x|mu) * log (p(x|mu)/ p(x))
			# loss += p(mu) * inner_sum