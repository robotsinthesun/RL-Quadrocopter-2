import numpy as np
import gym
from gym import spaces
from physics_sim import PhysicsSim

class TaskHover():

    def __init__(   self,
                    positionStart=None,
                    positionStartNoise=0,
                    positionTarget=None,
                    nActionRepeats= 1,
                    runtime=10.,
                    factorPositionZ=0.15,
                    factorPositionXY=0.0,
                    factorVelocityXY=0.3,
                    factorAngles=0.5,
                    factorAngleRates=0.0,
                    factorAngleAccels=0.0,
                    factorActions=0.05,
                    factorGlobal=1.0,
                    angleNoise=0,
                    angleRateNoise=0 ):

        # Internalize parameters.
        # Positions
        self.positionTarget = positionTarget if positionTarget is not None else np.array([0., 0., 10.])
        self.positionStart = positionStart if positionStart is not None else np.array([0., 0., 0.])
        # Reward factors.
        self.factorPositionZ = factorPositionZ
        self.factorPositionXY = factorPositionXY
        self.factorAngles = factorAngles
        self.factorAngleRates = factorAngleRates
        self.factorAngleAccels = factorAngleAccels
        self.factorActions = factorActions
        self.factorGlobal = factorGlobal
        self.positionStartNoise = positionStartNoise
        self.angleNoise = angleNoise
        self.angleRateNoise = angleRateNoise
        self.factorVelocityXY = factorVelocityXY
        # Number of action repeats.
        # For each agent time step we step the simulation multiple times and stack the states.
        self.nActionRepeats = nActionRepeats

        # Define action and state spaces.
        # Use OpenAI gym spaces to make this task behave
        # like an gym environment so we can use the same
        # agent as for the pendulum.
        # Init action space. Two actions, one for each side of the copter.
        # Limit actions to a range around hovering.
        self.action_low = 390
        self.action_high = 440
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(2,))
        # Number of action repeats.
        # For each agent time step we step the simulation multiple times.
        self.nActionRepeats = nActionRepeats
        # Here the state is comprised of target vector XZ, velocity XY, cos theta, sin theta and theta angular velocities (7 values).
        # Positions are bounded by the environment bounds, angles by +-pi and angular velocities by 40 rad/s (happens with two rotors 0 and two 900)
        # Because we step multiple times we receive a multiple of the state vector.
        #low = np.tile(np.hstack((self.sim.lower_bounds[[0,2]], self.sim.lower_bounds, np.array([0, 0, 0]), np.array([-np.pi*10.0, -np.pi*10.0, -np.pi*10.0]))), self.nActionRepeats)
        #high = np.tile(np.hstack((self.sim.upper_bounds, self.sim.upper_bounds, np.array([np.pi*2.0, np.pi*2.0, np.pi]), np.array([np.pi*10.0, np.pi*10.0, np.pi*10.0]))), self.nActionRepeats)
        low = np.tile(np.hstack(([-10, -10], [-20, -20], -1, -1, -40)), self.nActionRepeats)
        high = np.tile(np.hstack(([10, 10], [20, 20], 1, 1, 40)), self.nActionRepeats)
        self.observation_space = spaces.Box(low=low, high=high)


        # Init initial conditions.
        # Rotation is randomized by +-0.01 degrees around all axes.
        init_pose = np.hstack((self.positionStart, np.array([0., 0., 0.])))
        # Initial velocity 0,0,0.
        init_velocities = np.array([0., 0., 0.])
        # Initial angular velocity 0,0,0.
        init_angle_velocities = np.array([0., 0., 0.])

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)

    def get_reward(self, rotor_speeds):
        return self.rewardVelocityXY(self.sim.v) + self.rewardAction(rotor_speeds) + self.rewardPositionXY(self.sim.pose) + self.rewardPositionZ(self.sim.pose) + self.rewardAngles(self.sim.pose, self.sim.angular_v) + self.rewardAngleRates(self.sim.pose[3:6], self.sim.angular_v) + self.rewardAngleAccels(self.sim.angular_v, self.sim.angular_accels)

    def rewardPositionZ(self, pose):
        #cGauss = 2
        cExp = -0.2
        #yGauss = np.exp(-((pose[2]-self.positionTarget[2])**2/(2*cGauss**2)))
        yExp = np.exp(cExp * np.abs(self.positionTarget[2] - pose[2]))
        return self.factorPositionZ * yExp * self.factorGlobal

    def rewardPositionXY(self, pose):
        a = .5
        return self.factorPositionXY * np.exp(-np.power(np.linalg.norm(self.positionTarget[:2] - pose[:2]), a)) * self.factorGlobal

    def rewardAngles(self, pose, angular_v):
        angles = pose[3:5]
        # convert angles from 0--2*pi to -pi--pi
        # get max angle from all 3.
        angles = np.max(np.abs(  [a if a <= np.pi else a-2*np.pi for a in pose[3:5]]  ))
        #x = np.max(  np.abs(  [a if a <= np.pi else a-2*np.pi for a in pose[3:5]]  )   )
        #return np.cos(x) * self.factorAngles

        # Also, only give reward for small angles if the angular velocity is small as well.
        # Thereby we can avoid to give reward if the copter is tumbling.
        # We do this by
        cV=.8
        factorV = np.max(np.abs(angular_v[0:2]))
        factorV = np.exp(-((factorV**2)/(2*cV**2)))

        cA=0.5
        return np.exp(-((angles**2)/(2*cA**2))) * factorV * self.factorAngles * self.factorGlobal

    def rewardVelocityXY(self, v):
        c = 4
        vNorm = np.linalg.norm(v)
        return (np.exp(-((vNorm**2)/(2*c**2)))) * self.factorVelocityXY * self.factorGlobal

    def rewardAngleRates(self, angles, angular_v):
        '''
        cV=0.5
        factorV = np.max(np.abs(angular_v[0:2]))
        return (np.exp(-((x**2)/(2*c**2)))*2-1) * self.factorAngleRates
        '''
        #return np.tanh(np.sum(-np.sign([a if a <= np.pi else a-2*np.pi for a in angles[0:2]]) * angular_v[0:2])/2.) * self.factorAngleRates
        rewards = -np.tanh([a if a <= np.pi else a-2*np.pi for a in angles[0:2]]) * np.tanh(angular_v[0:2])
        rewardMax = rewards[np.argmax(np.abs(rewards))]
        return rewardMax * self.factorAngleRates * self.factorGlobal

    def rewardAngleAccels(self, angular_v, angular_accels):
        # Only reward acceleration if it reduces absolute angle rate.
        # If angle rate is positive, a negative acceleration shall be rewarded and vice versa.
        return np.tanh(np.sum(-np.sign(angular_v[0:2]) * angular_accels[0:2])/2.) * self.factorAngleAccels * self.factorGlobal

    def rewardAction(self, rotor_speeds):
        return -(np.mean(rotor_speeds)-self.action_low) / (self.action_high-self.action_low)  * self.factorActions

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.nActionRepeats):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) / self.nActionRepeats
            vectorTarget = self.positionTarget[[0,2]]-self.sim.pose[[0,2]]
            distanceTarget = np.max((np.linalg.norm(vectorTarget),1e-10))
            pose_all.append((vectorTarget/distanceTarget * np.min((10.0, distanceTarget))))
            pose_all.append(self.sim.v[[0,2]])# / np.max((np.linalg.norm(self.sim.v[[0,2]]),1e-10)) * np.min((20.0, np.linalg.norm(self.sim.v[[0,2]]))))
            #pose_all.append([self.sim.pose[4] if self.sim.pose[4] <= np.pi else self.sim.pose[4]-2*np.pi])
            pose_all.append([np.cos(self.sim.pose[4])])
            pose_all.append([np.sin(self.sim.pose[4])])
            pose_all.append([np.max((np.min((self.sim.angular_v[1], np.pi)), -np.pi))])
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, None

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset(positionNoise=self.positionStartNoise, angleNoise=self.angleNoise, angleRateNoise=self.angleRateNoise)
        vectorTarget = self.positionTarget[[0,2]]-self.sim.pose[[0,2]]
        distanceTarget = np.max((np.linalg.norm(vectorTarget),1e-10))
        state = np.concatenate([    (vectorTarget/distanceTarget * np.min((10.0, distanceTarget))),
                                    self.sim.v[[0,2]],# / np.max((np.linalg.norm(self.sim.v[[0,2]]),1e-10)) * np.min((20.0, np.linalg.norm(self.sim.v[[0,2]]))),
                                    #[self.sim.pose[4] if self.sim.pose[4] <= np.pi else self.sim.pose[4]-2*np.pi],
                                    [np.cos(self.sim.pose[4])],
                                    [np.sin(self.sim.pose[4])],
                                    [self.sim.angular_v[1]]   ] * self.nActionRepeats)
        return state
