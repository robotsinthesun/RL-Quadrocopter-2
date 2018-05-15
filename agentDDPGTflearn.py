"""
DDPG algorithm base on the DDPG code by Patrick Emami (https://github.com/pemami4911/deep-rl/tree/master/ddpg)
"""
import tensorflow as tf
import numpy as np
import gym
import tflearn

from actorTflearn import Actor
from criticTflearn import Critic
from ouNoiseTflearn import OUNoise
from replayBufferTflearn import ReplayBuffer


class AgentDDPG():


    def __init__(self, sess, env, hp):

        self.sess = sess
        self.env = env
        self.hp = hp
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low

        #self.state_size = self.env.state_size
        #self.action_size = self.env.action_size
        #self.action_low = self.env.action_low
        #self.action_high = self.env.action_high

        # Noise process
        self.noise = OUNoise(   mu=self.hp['noiseMu']*np.ones(self.action_size),
                                                    theta=self.hp['noiseTheta'],
                                                    sigma=self.hp['noiseSigmaStart'])

        # Replay memory
        self.memory = ReplayBuffer(int(self.hp['bufferSize']), int(self.hp['randomSeed']))


        # Algorithm parameters
        self.dropoutActor = 0.1
        self.dropoutCritic = 0.1

        np.random.seed(int(self.hp['randomSeed']))

        self.rewardSum = 0

        # Build networks.
        self.actor = Actor(self.sess, self.state_size, self.action_size, self.action_low, self.action_high, float(self.hp['actorLearningRate']), float(self.hp['tau']), int(self.hp['batchSize']))

        self.critic = Critic(self.sess, self.state_size, self.action_size,
                               float(self.hp['criticLearningRate']), float(self.hp['tau']),
                               float(self.hp['gamma']),
                               self.actor.get_num_trainable_vars())

        # Init networks.
        self.sess.run(tf.global_variables_initializer())
        self.actor.update_target_network()
        self.critic.update_target_network()



    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, (1, self.actor.s_dim))
        action = self.actor.predict(state)
        noise = self.noise.sample()
        return action + noise, noise  # add some noise for exploration

    

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(    np.reshape(self.last_state, (self.actor.s_dim,)),
                            np.reshape(action, (self.actor.a_dim,)),
                            reward,
                            done,
                            np.reshape(next_state, (self.actor.s_dim,)))

        # Accumulate reward.
        self.rewardSum += reward

        # Learn. Only if there are enough samples for a mini batch.
        if self.memory.size() > int(self.hp['batchSize']):
            # Get batch.
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.memory.sample_batch(int(self.hp['batchSize']))

            # Calculate targets.
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(int(self.hp['batchSize'])):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (int(self.hp['batchSize']), 1)))

            self.avgMaxQ += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Soft update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

        # Save last state.
        self.last_state = next_state



    def reset_episode(self):
        self.rewardSum = 0
        self.avgMaxQ = 0
        self.noise.reset()
        state = self.env.reset()
        self.last_state = state
        return state


