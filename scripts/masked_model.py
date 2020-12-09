import tensorflow as tf
import numpy as np

def mask(input, lmb=1):
  """
  input: tensor of shape (btch_size, x, y, channels)
  lmb: scaling factor for mask (the smaller lmb, the larger the mask)

  returns: Tensor masks of size (btch_size, x, y, channels)
  """
  assert input.shape[1] == input.shape[2], "Spectrograms not squares"

  btch_sz = input.shape[0]
  channels = input.shape[3]
  height_width = input.shape[1]
  
  np_input = input.numpy()
  np_input = np.swapaxes(np_input,1,3)
  np_input = np.swapaxes(np_input,2,3)
  
  #Referenced stack overflow for this function (https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop)
  def argmax_coord(A, N):
    s = A.shape
    new_shape = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shape).argmax(-1)
    return np.unravel_index(max_idx, s[-N:])

  coordinates = argmax_coord(np_input, 2)
  
  #Referenced public implementation for this portion of code (https://github.com/andrehuang/InterpretableCNN)
  mu_x = np.reshape(np.array(coordinates[0]), (btch_sz, 1, 1, channels))
  mu_y = np.reshape(np.array(coordinates[1]), (btch_sz, 1, 1, channels))

  mu_x = mu_x/((height_width-1.0)/2.0) - 1.0
  mu_y = mu_y/((height_width-1.0)/2.0) - 1.0
  temp_x = np.reshape(np.linspace(-1, 1, height_width), (1, height_width, 1, 1))
  temp_y = np.reshape(np.linspace(-1, 1, height_width), (1, 1, height_width, 1))
  
  pos_temp_x = np.tile(temp_x, (btch_sz, 1, height_width, channels))
  pos_temp_y = np.tile(temp_y, (btch_sz, height_width, 1, channels))
  
  mu_x = np.tile(mu_x, (1, height_width, height_width, 1))
  mu_y = np.tile(mu_y, (1, height_width, height_width, 1))

  mask = np.absolute(pos_temp_x - mu_x)
  mask = np.add(mask, np.absolute(pos_temp_y - mu_y))
  mask = np.maximum(1 - np.multiply(mask, lmb), -1)

  return tf.convert_to_tensor(mask)

class MaskedCNN(tf.keras.Model):
    def __init__(self, num_categories):
        """
        CNN with masking
        """
        super(MaskedCNN, self).__init__()

        # Defining hyperparameters and optimizer
        self.batch_sz = 20
        self.learn_rate = 0.001
        self.dropout_rate = 0.4
        self.hlayer_sz = 1024
        self.optimizer = tf.keras.optimizers.Adam(self.learn_rate)

        self.num_categories = num_categories

        # NOTE: use_bias for Conv2D is 'true' on default
        # Define model layers for convolution
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=2)

        # Define model layers for feed forward
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(self.hlayer_sz, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(10, activation=None)



    def call(self, inputs, labels):
        """
        Run forward pass on input batch of spectrograms
        :param inputs: batch of images, shape [batch_sz, 128, 128, 1]; labels: category of images, shape [batch_size, 1]
        :return: tuple of: (logits, with shape [batch_sz, 10]; conv2_outputs; and max categories for each filter)
        """
        # NOTE: there is only one channel for input images
        conv1_out = self.conv1(inputs)
        conv1_pool = self.pool1(conv1_out)
        conv2_out = self.conv2(conv1_pool)

        conv2_masked = tf.cast(conv2_out, tf.float32) * tf.cast(mask(conv2_out, lmb=0.5), tf.float32)
        conv2_pool = self.pool2(tf.keras.activations.relu(conv2_masked, alpha=0.0, max_value=None, threshold=0))

        # Flatten tensor to pass through linear layer(s)
        flattened = self.flatten(conv2_pool)
        dense_out = self.dense1(flattened)
        dropout = self.dropout(dense_out)
        logits = self.dense2(dropout)  # Shape [batch_sz, 10]

        #Calculate each filter's correct category
        batch_sz, h, w, channels = conv2_out.shape
        np_labels = labels.numpy()
        np_activations = conv2_out.numpy()

        activations = np.zeros((self.num_categories, h, w, channels))
        num_per_category = np.zeros((self.num_categories))

        for btch in range(batch_sz):
          category = np_labels[btch]
          activations[category, :, :, :] = activations[category, :, :, :] + np_activations[btch, :, :, :]
          num_per_category[category] = num_per_category[category] + 1

        sum_activations = np.sum(activations, axis=(1,2))
        average_activations = sum_activations/np.reshape(num_per_category, (self.num_categories, 1))
        filter_categories = np.argmax(average_activations, axis=0)

        # NOTE: compute softmax on logits (and find classes) in loss func
        print(list(filter_categories))
        return logits, conv2_pool, list(filter_categories)

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
