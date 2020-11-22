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
		self.conv1 = tf.keras.layers.Conv2D(32, (5,5), (1,1), 'same', activation='relu')
		self.pool1 = tf.keras.layers.MaxPool2D((2,2), strides=2)
		self.conv2 = tf.keras.layers.Conv2D(64, (5,5), (1,1), 'same', activation='relu')
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
		flattened = self.flat(conv2_pool)
		print(f"flat shape: {tf.shape(flattened)}")
		dense_out = self.dense1(flattened)
		dropout = self.dropout(dense_out)
		logits = self.dense2(dropout) # Shape [batch_sz, 10]

		# NOTE: compute softmax on logits (and find classes) in loss func
		return logits
    

	def loss_func(self, logits, labels):
		"""
		Calculates loss for basic CNN architecture
		:param logits: raw predictions of shape [batch_sz, 10] 
		"""
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
		
		return loss

	def accuracy_func(self, logits, lables):
		"""
		Calculates model's prediction accuracy -- to be used only on test set
		"""
		pass