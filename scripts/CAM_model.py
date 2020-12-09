import tensorflow as tf
import numpy as np

class CAMCNN(tf.keras.Model):
    def __init__(self):
        """
        CNN with global average pooling
        """
        super(CAMCNN, self).__init__()

        # Defining hyperparameters and optimizer
        self.batch_sz = 20
        self.learn_rate = 0.005
        self.dropout_rate = 0.4
        self.hlayer_sz = 1024
        self.optimizer = tf.keras.optimizers.Adam(self.learn_rate)

        # NOTE: use_bias for Conv2D is 'true' on default
        # Define model layers for convolution
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu')
        self.conv1a = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.conv2a = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=1, padding='same')

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.conv3a = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D((2, 2), strides=1)

        self.tconv1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))
        self.glob_avg1 = tf.keras.layers.GlobalAveragePooling2D()

        # Define model layers for feed forward
        self.dense1 = tf.keras.layers.Dense(10)
        #self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)


    def call(self, inputs):
        """
        Run forward pass on input batch of spectrograms
        :param inputs: batch of images, shape [batch_sz, 128, 128, 1]; labels: category of images, shape [batch_size, 1]
        :return: tuple of: (logits, with shape [batch_sz, 10]; conv2_outputs; and max categories for each filter)
        """
        # NOTE: there is only one channel for input images
        conv1_out = self.conv1(inputs)
        conv1a_out = self.conv1a(conv1_out)
        conv1_pool = self.pool1(conv1a_out)

        conv2_out = self.conv2(conv1_pool)
        conv2a_out = self.conv2a(conv2_out)
        conv2_pool = self.pool2(conv2a_out)

        conv3_out = self.conv3(conv2_pool)
        conv3a_out = self.conv3a(conv3_out)
        conv3_pool = self.pool3(conv3a_out)
        #print(conv3_pool.shape)

        tconv1_out = self.tconv1(conv3_out)
        print(tconv1_out.shape)
        glob_avg1_out = self.glob_avg1(tconv1_out)

        logits = self.dense1(glob_avg1_out)  # Shape [batch_sz, 10]

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
