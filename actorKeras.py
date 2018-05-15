from keras import layers, models, optimizers, initializers
from keras import backend as K
from keras.utils import plot_model

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, architecture, activation, batchNormalization, learningRate, dropout, dropoutRate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.architecture = architecture
        self.batchNormalization = batchNormalization
        self.activation = activation
        self.learningRate = learningRate
        self.dropout = dropout
        self.dropoutRate = dropoutRate
        
        

        # Initialize any other variables here

        self.build_model()
       

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""

        # Set random uniform initialization with [-3e-3, 3e-3] to have all weights set around 0.
        initSeed = 0
        initMin = -3e-3
        initMax = 3e-3
        
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers according to network architecture.
        for i, nUnits in enumerate(self.architecture):
            net = layers.Dense(units=nUnits)(net if i>0 else states)
            if self.batchNormalization:
                net = layers.BatchNormalization()(net)
            if self.activation == 'leakyRelu':
                net = layers.LeakyReLU(alpha=.001)(net)
            else:
                net = layers.Activation(activation=self.activation)(net)
            if self.dropout:
                net = layers.Dropout(self.dropoutRate)(net)

        # Add final output layer with sigmoid activation
        net = layers.Dense(units=self.action_size, kernel_initializer=initializers.RandomUniform(minval=initMin, maxval=initMax, seed=initSeed))(net)
        if self.batchNormalization:
                net = layers.BatchNormalization()(net)
        raw_actions = layers.Activation(activation='sigmoid', name='raw_actions')(net)

        # Note that the raw actions produced by the output layer are in
        # a [0.0, 1.0] range (using a sigmoid activation function).
        # So, we add another layer that scales each output to the desired range
        # for each action dimension. This produces a deterministic action for
        # any given state vector. A noise will be added later to this action
        # to produce some exploratory behavior.
        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)
        
        
        plot_model(self.model, to_file='actorKeras.png', show_shapes=True)

        # Define the loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learningRate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        # The Q-value gradient will need to be computed using the critic model, and fed
        # in while training. Hence it is specified here as part of the "inputs" used in the training function.
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()], # WHAT IS THIS?
            outputs=[],
            updates=updates_op)

