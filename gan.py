# Jan Kremer, 2017
# Tensorflow implementation of the improved Wasserstein GAN by Gulrajani et al., 2017, https://arxiv.org/abs/1704.00028
# based on the blog posts by Eric Jang, http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
# and by John Glover, http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
# and the Wasserstein-GP implementation at https://github.com/igul222/improved_wgan_training

import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib.keras import layers


class GAN(object):
    """Implementation of the WGAN-GP algorithm.

    The models for critic and generator are relatively simple and can be modified for anything more complicated than
    the 1D toy example.
    """

    def __init__(self, n_step=2000, n_critic=5, n_batch=64, n_hidden=4, n_sample=10000, learning_rate=1e-3,
                 lambda_reg=0.1, log_interval=10, seed=0, beta1=0.5, beta2=0.9, verbose=True, callback=None):
        """Initialize the GAN.

        :param n_step: Number of optimization steps.
        :param n_critic: Number of critic optimization step per generator optimization step.
        :param n_batch: Mini-batch size.
        :param n_hidden: Number of hidden neurons in critic and generator.
        :param n_sample: Number of samples to draw from the model.
        :param learning_rate: The learning rate of the optimizer.
        :param lambda_reg: The regularization parameter lambda that controls the gradient regularization when training.
        :param log_interval: The number of steps between logging the training process.
        :param seed: The seed to control random number generation during training.
        :param beta1: Hyperparameter to control the first moment decay of the ADAM optimizer.
        :param beta2: Hyperparameter to control the second moment decay of the ADAM optimizer.
        :param verbose: Whether to print log messages during training or not.
        :param callback: Callback method to call after each training step with signature (model, session, data).
        """
        self.n_step = n_step
        self.n_critic = n_critic
        self.n_batch = n_batch
        self.n_hidden = n_hidden
        self.n_sample = n_sample
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.log_interval = log_interval
        self.seed = seed
        self.beta1 = beta1
        self.beta2 = beta2
        self.verbose = verbose
        self.callback = callback
        self.loss_curve = []
        self.graph = self._create_graph()

    def _create_generator(self, activation=tf.nn.relu):
        """Create the computational graph of the generator and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h = layers.Dense(self.n_hidden, activation=activation)
        output = layers.Dense(1)
        return lambda x: output(h(x))

    def _create_critic(self, activation=tf.nn.relu):
        """Create the computational graph of the critic and return it as a functional of its input.

        :param activation: The activation function to use.
        :return: Functional to create the tensorflow operation given its input.
        """
        h = layers.Dense(self.n_hidden, activation=activation)
        output = layers.Dense(1)
        return lambda x: output(h(x))

    def _create_optimizer(self, loss, var_list, learning_rate, beta1, beta2):
        """Create the optimizer operation.

        :param loss: The loss to minimize.
        :param var_list: The variables to update.
        :param learning_rate: The learning rate.
        :param beta1: First moment hyperparameter of ADAM.
        :param beta2: Second moment hyperparameter of ADAM.
        :return: Optimizer operation.
        """
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(loss, var_list=var_list)

    def _create_graph(self):
        """Creates the computational graph.

        :return: The computational graph.
        """
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)  # Fix the random seed for randomized tensorflow operations.

            with tf.variable_scope('generator'):  # Create generator operations.
                self.z = tf.placeholder(tf.float32, shape=(None, 1))
                self.G = self._create_generator()(self.z)

            with tf.variable_scope('critic'):  # Create critic operations.
                self.x = tf.placeholder(tf.float32, shape=(None, 1))
                D = self._create_critic()
                self.D_real = D(self.x)  # Criticize real data.
                self.D_fake = D(self.G)  # Criticize generated data.

                # Create the gradient penalty operations.
                epsilon = tf.random_uniform(shape=tf.shape(self.x), minval=0., maxval=1.)
                interpolation = epsilon * self.x + (1 - epsilon) * self.G
                penalty = (tf.norm(tf.gradients(D(interpolation), interpolation)) - 1) ** 2.0

            # Create the loss operations of the critic and generator.
            self.loss_d = -tf.reduce_mean(self.D_real - self.D_fake) + self.lambda_reg * penalty
            self.loss_g = -tf.reduce_mean(self.D_fake)

            # Store the variables of the critic and the generator.
            self.vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
            self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

            # Create optimizer operations for critic and generator.
            self.opt_d = self._create_optimizer(self.loss_d, self.vars_d, self.learning_rate, self.beta1, self.beta2)
            self.opt_g = self._create_optimizer(self.loss_g, self.vars_g, self.learning_rate, self.beta1, self.beta2)

            # Create variable initialization operation.
            self.init = tf.global_variables_initializer()
            graph.finalize()
        return graph

    def _sample_latent(self, n_sample):
        """Sample the input data to generate synthetic samples from.

        :param n_sample: Sample size.
        :return: Sample of input noise.
        """
        return np.random.randn(n_sample, 1)

    def fit(self, X):
        """Fit the GAN model.

        :param X: Training data.
        :return: The fit model.
        """
        np.random.seed(self.seed)  # Fix the seed for random data generation in numpy.
        with tf.Session(graph=self.graph) as session:
            start = time.time()
            session.run(self.init)
            for step in range(self.n_step + 1):
                # Optimize the critic for several rounds.
                for _ in range(self.n_critic):
                    x, _ = X.next_batch(self.n_batch)
                    z = self._sample_latent(self.n_batch)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], {self.x: x, self.z: z})

                # Sample noise and optimize the generator.
                z = self._sample_latent(self.n_batch)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {self.z: z})

                # Log the training procedure and call callback method for actions like plotting.
                if step % self.log_interval == 0:
                    self.loss_curve += [-loss_d]
                    if self.verbose:
                        elapsed = int(time.time() - start)
                        print('step: {:4d}, negative critic loss: {:8.4f}, time: {:3d} s'.format(step, -loss_d, elapsed))
                    if self.callback is not None:
                        self.callback(self, session, X)
        return self

    def sample(self, session):
        """Sample generated data.

        :param session: The current tensorflow session holding the trained graph.
        :return: A sample of generated data.
        """
        z = self._sample_latent(self.n_sample)
        return np.array(session.run(self.G, {self.z: z}))

    def critic(self, session, x):
        """Returns the critic function.

        :param session: Tensorflow session.
        :param x: Input data to criticize.
        :return: The current critic function.
        """
        return np.array(session.run(self.D_real, {self.x: x}))
