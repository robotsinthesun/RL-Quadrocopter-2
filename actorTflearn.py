# Modified from http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

import tensorflow as tf
import numpy as np
import tflearn


class Actor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_low, action_high, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        
        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]
        
        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
        
        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        
        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))
        
        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)
        
    def create_actor_network(self):
        # Random uniform initializer with range of [-1/sqrt(fanIn), 1/sqrt(fanIn)]
        w_init_hidden = tflearn.initializations.variance_scaling(factor=1.0, mode='FAN_IN', uniform=True)
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, weights_init=w_init_hidden)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300, weights_init=w_init_hidden)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.add(tf.multiply(tf.add(tf.multiply(out, 0.5), 0.5), self.action_high-self.action_low), self.action_low)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
