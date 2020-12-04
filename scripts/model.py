import tensorflow as tf
import tensorflow.keras.backend as kb
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
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), (1, 1), "same", activation="relu"
        )
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), (1, 1), "same", activation="relu"
        )
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=2)

        # Define model layers for feed forward
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.hlayer_sz, activation="relu")
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
        # mask_func(conv2_out)
        conv2_pool = self.pool2(conv2_out)

        # Flatten tensor to pass through linear layer(s)
        flattened = self.flatten(conv2_pool)
        dense_out = self.dense1(flattened)
        dropout = self.dropout(dense_out)
        logits = self.dense2(dropout)  # Shape [batch_sz, 10]

        # NOTE: compute softmax on logits (and find classes) in loss func
        return logits

    def mask_func(self, maps):
        """
		Uses feature maps select and apply desired template
		:param maps: 64 feature maps outputted from higher conv2D layer
		:returns: masked feature maps
		"""
        # TODO: write mask function in TF ops
        pass

    def loss_func(self, logits, labels):
        """
		Calculates loss for basic CNN architecture
		:param logits: raw predictions of shape [batch_sz, 10] 
		:param labels: correct labels for given batch [batch_sz]
		:returns: final task loss as tf scalar
		"""
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

        return tf.reduce_mean(loss)

    def accuracy_func(self, logits, labels):
        """
		Calculates model's prediction accuracy
		:param logits: raw predictions of shape [batch_sz, 10]
		:param labels: correct labels for given batch [batch_sz]
		:returns: batch accuracy as tf scalar
		"""
        num_correct_predictions = tf.equal(tf.argmax(logits, 1), labels)

        return tf.reduce_mean(tf.cast(num_correct_predictions, tf.float32))

    # NOTE: we will not be using the interpretable loss and instead implementing global average pooling for interpretability
    # def filter_loss_func(self, maps, filter_categories, input_categories, templates):
    # 	"""
    # 	Calculates filter loss in forward pass
    # 	:param maps: feature maps after 2nd Conv2D layer and ReLU op [batch_sz, num_rows, num_cols, 64]
    # 	:param filter_categories: assigned category for each filter [64]
    # 	:returns: filter losses
    # 	"""
    # 	# Convert inputs to numpy objects for easier manipulation
    # 	maps = maps.numpy()
    # 	filter_categories = filter_categories.numpy()
    # 	batch_sz = np.shape(maps)[0]

    # 	# Define "hyperparameter(s)"
    # 	num_rows = 0
    # 	num_cols = 0
    # 	tau = 0.5/(num_rows)^2
    # 	alpha = (num_rows^2)/(1+num_rows^2)

    # 	# Define true hyperparameter(s)
    # 	beta = 4

    # 	filter_losses = [] # List of filter losses for each filter for each input image
    # 	complete_template_mu_list = [] # List of every template for every mu for every feature map for every image

    # 	# Iterate through each image in batch
    # 	for input_num in range(batch_sz):
    # 		# Extract input map for specific image and reshape to [64, num_rows, num_cols]
    # 		input_map = np.reshape(np.squeeze(maps[input_num, :, :, :]), (64, num_rows, num_cols))

    # 		# Iterate through each map
    # 		for i, x in zip(range(len(input_map)), input_map):
    # 			# Determine target part location
    # 			target_part = np.argmax(x)

    # 			# x fits to target template
    # 			if target_part == filter_categories[i]:
    # 				template_mu = np.zeros((num_rows, num_cols)) # Filled at runtime
    # 				p_mu = alpha / num_rows^2
    # 				is_negative = False
    # 			# x fits to negative template
    # 			else:
    # 				template_mu = np.full_like((num_rows, num_cols), fill_value=-tau)
    # 				p_mu = 1 - alpha
    # 				is_negative = True

    # 			# Calculate templates for every mu in feature map x
    # 			template_mu_list = [] # Will be of shape [num_rows^2]
    # 			for mu in range(len(num_rows^2)):
    # 				# Calculate positive template for every mu, if x fit to target template
    # 				if not is_negative:
    # 					# Compute L1 norm distance
    # 					for i in range(num_rows):
    # 						for j in range(num_cols):
    # 							l1_norm = np.linalg.norm(np.array([i,j]) - mu, ord=1)
    # 							template_mu[i][j] = -tau * np.maximum(1 - beta * (l1_norm / num_rows), -1)
    # 					template_mu_list.append(template_mu)
    # 				# Append negative template for every mu, if x didn't fit to target template
    # 				else:
    # 					template_mu_list.append(template_mu)

    # 			# Iterate through each part location of each feature map to determine loss
    # 			for mu in range(len(num_rows^2)):
    # 				inner_sum = 0
    # 				z_mu = 0

    # 				# Get template
    # 				template_mu = template_mu_list[mu]

    # 				# Calculate z_mu -> constant to compute gradients
    # 				for x in input_map:
    # 					z_mu += np.exp(np.matmul(x, template_mu))

    # 				# Calculate p(x|mu) -> fitness between feature map and part location
    # 				for x in input_map:
    # 					p_x_mu = (1/z_mu) * np.exp(np.matmul(x, template_mu))
    # 					# Calculate p(x) -> probability of a feature map
    # 					for all mu in range(len(num_rows^2)):
    # 						p_x += p_mu * ((1/z_mu) * np.exp(np.matmul(x, template_mu_list[mu])))
    # 					inner_sum += p_x_mu * np.log(p_x_mu / p_x)

    # 				# Sum filter loss for each mu
    # 				filter_loss += p_mu * inner_sum

    # 			# Append all templates for map to complete_template_mu_list [64, num_rows^2]
    # 			complete_template_mu_list.append(template_mu_list)

    # 			# Add filter loss for feature map
    # 			input_map[i] = np.add(x, -filter_loss)
    # 			filter_losses.append(-filter_loss)

    # # Reshape filter losses to [batch_sz, 64]
    # filter_losses = tf.convert_to_tensor(np.reshape(filter_losses, (batch_sz, 64)))

    # # Reshape all templates to [batch_sz, 64, num_rows^2]
    # complete_template_mu_list = tf.convert_to_tensor(np.reshape(complete_template_mu_list, (batch_sz, 64, num_rows^2)))
    # return filter_losses

