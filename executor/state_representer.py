import os
import random
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Activation, Dense, Reshape

from Executor.utils import get_executor_config, R_DIM, S_DIM, Path, DEFAULT_DEVICE, N_FRAMES, TARGET_R_SHAPE

ALPHA = 1.
BETA = 2.
DROPOUT_RATE = .1
H, W = 100, 100


class VariantionalAutoEncoder(object):
    def __init__(self, sess, learning_rate=1e-4, batch_size=64, scope='compressor', max_buff_size=5*10 ** 5):
        self.sess = sess
        self.max_buff_size = max_buff_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = R_DIM
        self.n_in = S_DIM * N_FRAMES
        self.in_shape = TARGET_R_SHAPE + [N_FRAMES]
        self.scope = scope
        self.Structure = get_executor_config('state_compressor')
        self.add_new_data = False
        flag = 'r+' if os.path.exists(Path + 'compressor_data.dat') else 'w+'
        self.data = np.memmap(Path + 'compressor_data.dat', dtype=np.float, mode=flag, shape=tuple([max_buff_size] + self.in_shape))
        self.Cursor = 0
        self.is_full = False
        self.weights: list = None
        self.learning = False
        self.build()

    def start_learning(self):
        self.learning = True

    def stop_learning(self):
        self.learning = False

    # Build the netowrk and the loss functions
    def build(self):
        encoder_conv_config = self.Structure[0]
        decoder_conv_config = self.Structure[1]
        dense_encoder = self.Structure[2]
        dense_decoder = self.Structure[3]

        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None] + self.in_shape)

        # Encode
        # x -> z_mean, z_sigma -> z
        X = self.x

        with tf.variable_scope(self.scope + '_compressor'):
            for filters, kernel, stride, padding in encoder_conv_config:
                X = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding=padding, activation='relu')(X)

        CONV_SHAPE = X.shape[1:]
        X = tf.contrib.layers.flatten(X)

        with tf.variable_scope(self.scope + '_compressor'):
            for units in dense_encoder:
                X = Dense(units, activation='relu')(X)

            self.z_mu = X
            self.z_log_var = Dense(self.n_z, activation='softplus')(X)

        eps = tf.random_normal(
            shape=tf.shape(self.z_log_var),
            mean=0, stddev=1, dtype=tf.float32)

        self.z = tf.nn.tanh(self.z_mu + eps * tf.exp(self.z_log_var * .5))

        # Decode
        # z -> x_hat
        Y = self.z

        with tf.variable_scope(self.scope + '_compressor'):
            for units in dense_decoder:
                Y = Dense(units, activation='relu')(Y)

        Y = Reshape(target_shape=CONV_SHAPE)(Y)

        with tf.variable_scope(self.scope + '_compressor'):
            for filters, kernel, stride, padding in decoder_conv_config:
                Y = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=stride, padding=padding,
                                    activation='relu')(Y)

            Y = Conv2DTranspose(filters=N_FRAMES, kernel_size=3, strides=(1, 1), padding="SAME")(Y)

        self.x_hat = Activation(activation='sigmoid')(Y)
        self.weights = list(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '_compressor'))

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.contrib.layers.flatten(self.x_hat),
                                                            labels=tf.contrib.layers.flatten(self.x))
        logpx_z = -tf.reduce_mean(cross_ent, axis=[0, 1])
        logpz = self.log_normal_pdf(self.z, 0., 0.)
        logqz_x = self.log_normal_pdf(self.z, self.z_mu, self.z_log_var)
        self.total_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi))

    def add_data(self, item):
        self.data[self.Cursor] = item
        self.Cursor += 1
        if self.Cursor >= self.max_buff_size:
            self.Cursor = 0
            self.is_full = True

    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.total_loss],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def compress_state(self, x):
        try:
            # x = np.load('C:\\Users\\Rubil\\PycharmProjects\\model_converter\\tmp.npy')

            feed_dict = {self.x: x.reshape([-1] + self.in_shape)}
            z = self.sess.run(self.z_mu, feed_dict=feed_dict)
            return z
        except Exception:
            return x

    def create_images(self, w, h, number=1):
        images_origin = self.sample(number)
        images = self.reconstructor(images_origin)
        for i in range(number):
            image, image_sample = images[i, :], images_origin[i, :]
            image = np.rint(image * 255)
            image_sample = np.rint(image_sample * 255)
            plt.imshow(image_sample.reshape(w, h), interpolation='nearest')
            plt.imshow(image.reshape(w, h), interpolation='nearest')

        plt.show()

    def create_correct_images(self, w, h, number=1):
        if self.is_full:
            images = random.choices(self.data, k=number)
        else:
            ids = [random.randint(0, self.Cursor) for _ in range(number)]
            images = self.data[ids]
        for i in range(number):
            image = images[i]
            image = np.rint(image * 255)
            plt.imshow(image.reshape(w, h), interpolation='nearest')
            plt.show()

    def sample(self, batch):
        max_id = self.max_buff_size if self.is_full else self.Cursor
        ids = [np.random.randint(0, max_id) for _ in range(batch)]
        S = [np.copy(self.data[i]) for i in ids]
        return np.array(S)

    def learn(self, iters=1):
        loss = 0.
        for t in range(iters):
            S = self.sample(self.batch_size)
            loss += self.run_single_step(S)
        return loss/iters

    def update_params(self, params):
        ops = []
        for param1, param2 in zip(params, self.weights):
            ops.append(tf.assign(param2, param1))

        with tf.device(DEFAULT_DEVICE):
            self.sess.run(ops)

    def get_params(self):
        return self.sess.run(self.weights)

    def save(self, path: str):
        W = self.sess.run(self.weights)
        np.save(path + 'compressor', W)

    def restore(self, path: str):
        W = np.load(path + 'compressor.npy', allow_pickle=True)
        ops = []

        for var, w in zip(self.weights, W):
            ops.append(tf.assign(var, w))
        self.sess.run(ops)

    def save_data(self):
        try:
            data = np.array([self.Cursor, self.is_full])
            np.save(Path + 'Cursor_compressor', data)
            print('Compressor data saved')
        except Exception:
            print('Compressor data has not bben saved')

    def restore_data(self):
        try:
            data = np.load(Path + 'Cursor_compressor.npy')
            self.Cursor = data.item(0)
            self.is_full = bool(data.item(1))
            print('Compressor data restored')
            return self.is_full
        except Exception:
            print('Compressor data is not restored')
            return False

    def clear_data(self):
        del self.data
