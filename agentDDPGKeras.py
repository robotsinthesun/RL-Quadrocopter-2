import numpy as np
from replayBufferKeras import ReplayBuffer
from actorKeras import Actor
from criticKeras import Critic
from ouNoiseKeras import OUNoise

# Note that we will need two copies of each model - one local and one target.
# This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning,
# and is used to decouple the parameters being updated from the ones
# that are producing target values.



class AgentDDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, hyperparams):
        print("Setting up agent.")

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.action_range = self.action_high - self.action_low

        self.rewardSum = 0

        # Actor (Policy) Model
        print("   Creating actor...", end='')
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.action_low,
                                 self.action_high,
                                 hyperparams['actorArch'],
                                 hyperparams['actorAF'],
                                 hyperparams['actorBN'],
                                 hyperparams['actorLR'],
                                 hyperparams['actorDO'],
                                 hyperparams['actorDORate'])
        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.action_low,
                                  self.action_high,
                                  hyperparams['actorArch'],
                                  hyperparams['actorAF'],
                                  hyperparams['actorBN'],
                                  hyperparams['actorLR'],
                                  hyperparams['actorDO'],
                                  hyperparams['actorDORate'])
        print(" Done.")
        #print(self.actor_target.model.summary())

        # Critic (Q-Value) Model
        print("   Creating critc...", end='')
        self.critic_local = Critic(self.state_size,
                                   self.action_size,
                                   hyperparams['criticArch'],
                                   hyperparams['criticAF'],
                                   hyperparams['criticBN'],
                                   hyperparams['criticLR'],
                                   hyperparams['criticDO'],
                                   hyperparams['criticDORate'])
        self.critic_target = Critic(self.state_size,
                                    self.action_size,
                                    hyperparams['criticArch'],
                                    hyperparams['criticAF'],
                                    hyperparams['criticBN'],
                                    hyperparams['criticLR'],
                                    hyperparams['criticDO'],
                                    hyperparams['criticDORate'])
        print(" Done.")
        #print(self.critic_target.model.summary())

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        print("   Creating noise model...", end='')
        # Mu is the mean of the noise.
        # Over time, the noise will slowly approach this mean.
        #self.exploration_mu = 0
        # Theta controls how
        #self.exploration_theta = 0.15
        #self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size,
                             hyperparams['noiseMu'],
                             hyperparams['noiseTheta'],
                             hyperparams['noiseSigmaStart'],
                             hyperparams['noiseSigmaStart']-hyperparams['noiseReduction'],
                             hyperparams['noiseDecay'])
        print(" Done.")

        # Replay memory
        print("   Creating replay buffer...", end='')
        self.buffer_size = hyperparams['replaySize']
        self.batch_size = hyperparams['replayBatch']
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        print(" Done.")

        # Algorithm parameters
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']

    def reset_episode(self):
        self.rewardSum = 0
        self.noise.reset()
        state = self.env.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.rewardSum += reward

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    #
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        #print(state)
        action = self.actor_local.model.predict(state)[0]
        noise = self.noise.sample()
        return list(action + noise), noise  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Here, we first train a local copy of the actor and critic models. Then,
        # we only copy a fraction of the new weights over to the real actor and critic
        # models (actor_target and critic_target). This fraction is defined by
        # parameter tau.

        # We need to compute a full SARSA tuple in order for actor and critic to learn.
        # The actor needs St, At and the Q-value Q(St, At) for it's update.
        # The critic needs St, At, Rt+1, St+1 and At+1 for it's update.
        # In the experience tuple we have all except for At+1.

        # First, convert experience SARS tuples to separate arrays for each element (states, actions, rewards, etc.).
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Then, predict the next-state actions At+1 and Q values from target models.
        # The actor predicts the next action from the current state.
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        # The critic calculates the next states Q-value Q(St+1, At+1).
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    # Update the weights.
    # Notice that after training over a batch of experiences,
    # we could just copy our newly learned weights (from the local model)
    # to the target model. However, individual batches can introduce a
    # lot of variance into the process, so it's better to perform a
    # soft update, controlled by the parameter tau.
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        # Compute the weighted average of local and target weights using tau.
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
