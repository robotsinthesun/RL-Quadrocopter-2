# The critic model is simpler than the actor model in some ways,
# but there some things worth noting.
# Firstly, while the actor model is meant to map states to actions,
# the critic model needs to map (state, action) pairs to their Q-values.
# These two inputs can first be processed via separate "pathways"
# (mini sub-networks), but eventually need to be combined.
# This can be achieved, for instance, using the Add layer
# type in Keras (see Merge Layers)

from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as K
from keras.utils import plot_model

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, architecture, activation, batchNormalization, learningRate, dropout, dropoutRate):
        # Initialize parameters and build model.

        self.state_size = state_size
        self.action_size = action_size
        self.architecture = architecture
        self.batchNormalization = batchNormalization
        print(self.batchNormalization)
        self.activation = activation
        self.learningRate = learningRate
        self.dropout = dropout
        self.dropoutRate = dropoutRate

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # Set random uniform initialization with [-3e-4, 3e-4] to have all weights set around 0.
        # This is according to DDPG paper.
        initSeed = 0
        initMin = -3e-3
        initMax = 3e-3
        l2Lambda = 1e-2
        
        # Define separate input layers for state and action pathways (sub-networks).
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        for i, nUnits in enumerate(self.architecture[0]):
            net_states = layers.Dense(units=nUnits, activity_regularizer=regularizers.l2(l2Lambda))(net_states if i>0 else states)
            # Add batch normalization, activation and dropout.
            if self.batchNormalization:
                net_states = layers.BatchNormalization()(net_states)
            if self.activation == 'leakyRelu':
                net = layers.LeakyReLU(alpha=.001)(net)
            else:
                net_states = layers.Activation(activation=self.activation)(net_states)
            if self.dropout:
                net_states = layers.Dropout(self.dropoutRate)(net_states)

        # Add hidden layer(s) for action pathway
        for i, nUnits in enumerate(self.architecture[1]):
            net_actions = layers.Dense(units=nUnits, activity_regularizer=regularizers.l2(l2Lambda))(net_actions if i>0 else actions)
            # Add batch normalization, activation and dropout.
            if self.batchNormalization:
                net_actions = layers.BatchNormalization()(net_actions)
            if self.activation == 'leakyRelu':
                net = layers.LeakyReLU(alpha=.001)(net)
            else:
                net_actions = layers.Activation(activation=self.activation)(net_actions)
            if self.dropout:
                net_actions = layers.Dropout(self.dropoutRate)(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        if self.activation == 'leakyRelu':
            net = layers.LeakyReLU(alpha=.001)(net)
        else:
            net = layers.Activation('relu')(net)
        if self.dropout:
            net = layers.Dropout(self.dropoutRate)(net)
        

        # Add more layers to the combined network if needed
        for nUnits in self.architecture[2]:
            net = layers.Dense(units=nUnits, activity_regularizer=regularizers.l2(l2Lambda))(net)
            if self.batchNormalization:
                net = layers.BatchNormalization()(net)
            if self.activation == 'leakyRelu':
                net = layers.LeakyReLU(alpha=.001)(net)
            else:
                net = layers.Activation(activation=self.activation)(net)
            if self.dropout:
                net = layers.Dropout(self.dropoutRate)(net)
                                   
        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, kernel_initializer=initializers.RandomUniform(minval=initMin, maxval=initMax, seed=initSeed), name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        plot_model(self.model, to_file='criticKeras.png', show_shapes=True)
        
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # The final output of this model is the Q-value for any given (state, action) pair.
        # However, we also need to compute the gradient of this Q-value with respect
        # to the corresponding action vector, needed for training the actor model.
        # This step needs to be performed explicitly, and a separate function needs
        # to be defined to provide access to these gradients.
        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
