import os
import numpy as np
import gymnasium as gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from collections import deque
import random
from get_data import get_data
from utils import get_memory

class StockTradingEnvironment(gym.Env):
    """Create a custom Gym environment
    data is formated as ["Open", "High", "Low", "Close", "Volume"]
    """
    def __init__(self, duration = 120, max_trade_len = 1000, penalty = 0.00001, min_gain = 0.001, max_gain = 0.05, min_loss = 0.001, max_loss = 0.05):
        super(StockTradingEnvironment, self).__init__()
        self._data = None
        self.duration = duration    # Time window of stock price to analyze
        self.total_steps = 0
        self.penalty = penalty      # penalty for staying idle
        self.min_gain = min_gain    # Min gain per trade
        self.max_gain = max_gain    # Max gain per trade
        self.min_loss = min_loss    # Min loss per trade
        self.max_loss = max_loss    # Max loss per trade
        self.action_dict = self.create_action_idx(bins_per_dim = 10) # create action dict, 1 action for not entering trade and many actions with entering trades
        self.action_space = gym.spaces.Discrete(len(self.action_dict))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(duration, 5), dtype=np.float32)

        self.current_step = 0
        self.position = None
        self.norm_factor = None
        self.enter_price = None
        self.max_trade_len = max_trade_len    # Max duration of a trade

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data
        self.total_steps = new_data.shape[0] - self.duration

    def reset(self):
        self.current_step = 0
        self.position = None
        return self.get_state()
    
    def get_state(self):
        state = self.data[self.current_step : self.current_step + self.duration]
        # normalize the state: Open, High, Low, Close are divided by max High; Volume is divided by max Volume
        self.norm_factor = tf.reduce_max(state[:, 1])
        volume_norm_factor = tf.reduce_max(state[:, 4])
        state = tf.concat([state[:, :4] / self.norm_factor, state[:, 4:] / volume_norm_factor], axis=1)
        state = tf.Variable(state, dtype=tf.float32)
        return state
    
    # Modify and use the following function to discretize continuous actions
    def create_action_idx(self, bins_per_dim):
        action_dict = {}
        action_dict[0] = [0, 0, 0]
        key = 1
        for gain in np.linspace(self.min_gain, self.max_gain, bins_per_dim + 1):
            for loss in np.linspace(self.min_loss, self.max_loss, bins_per_dim + 1):
                if gain >= loss:
                    action_dict[key] = [1, gain, loss]
                    key += 1
        return action_dict
    
    def step(self, action_idx):
        action = self.action_dict[action_idx]

        if self.position is None and action[0] == 1:
            self.position = 'long'
            self.enter_price = self.data[self.current_step + self.duration - 1, 3]  # enter price is the last "Close"
            self.enter_norm_factor = self.norm_factor
            # print(f"Entering trade at price: {self.enter_price.numpy()}")
        
        # give a negative reward every time step if doing nothing
        reward = - self.penalty
        
        # if a trade has been entered, find the exit and get the reward
        # exit is attained when: 1. gain is reached, 2. loss is reached, 3. max duration for trade is reached
        if self.position == 'long':
            price_ceiling = self.enter_price * (1 + action[1])
            price_floor = self.enter_price * (1 - action[2])

            if len(self.data) >= self.current_step + self.duration + self.max_trade_len:
                exit_window = self.data[self.current_step + self.duration - 1 : self.current_step + self.duration + self.max_trade_len]
            else:
                exit_window = self.data[self.current_step + self.duration - 1 :]

            exit_gain = tf.argmax(exit_window[:,1] >= price_ceiling).numpy()
            exit_loss = tf.argmax(exit_window[:,2] <= price_floor).numpy()
            if exit_loss == exit_gain == 0:
                time_passed = len(exit_window) - 1
            elif exit_loss == 0:
                time_passed = exit_gain
            elif exit_gain == 0:
                time_passed = exit_loss
            else:
                time_passed = np.min([exit_gain, exit_loss])

            self.current_step += time_passed
            next_state = self.get_state()
            _return = (next_state[-1, 3] * self.norm_factor / self.enter_price).numpy() - 1
            reward += _return - self.penalty * time_passed
            self.position = None
            # print(f"Exiting trade at price: {(next_state[-1, 3] * self.norm_factor).numpy()} at a gain/loss of {_return} after {time_passed} time steps")
        else:
            self.current_step += 1
            next_state = self.get_state()
            time_passed = 1

        done = self.current_step >= self.total_steps
        
        return next_state, reward, done, time_passed

class DQNAgent:
    """Create Deep Q Learning Agent
    """
    def __init__(self, state_shape, action_shape, epsilon=0.1, epsilon_decay=0.99, risk_averse=0.8, learning_rate=0.001, gamma=0.9999, tau=0.1, max_memory=10000):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.risk_averse = risk_averse
        self.gamma = gamma
        self.tau = tau

        self.model = self.build_dqn()
        self.target_model = self.build_dqn()
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_function = tf.keras.losses.Huber()

        self.memory = deque(maxlen=max_memory)  # Store experience replay

    def build_dqn(self):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=self.state_shape, name='conv1d_1'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=512, kernel_size=3, activation='relu', name='conv1d_2'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', name='dense_3'))
        model.add(Dense(512, activation='relu', name='dense_2'))
        model.add(Dense(self.action_shape, activation='linear'))
        return model

    def clean_memory(self):
        self.memory.clear()
        self.epsilon = self.init_epsilon

    def select_action(self, state):
        # select action randomly with prob of epsilon
        if np.random.uniform() < self.epsilon:
            if np.random.uniform() < self.risk_averse:
                return 0
            else:
                return np.random.choice(self.action_shape - 1) + 1
        q_values = self.model(state[None, ...])[0]
        return tf.argmax(q_values).numpy()

    def remember(self, state, action, reward, next_state, done, time_passed):
        self.memory.append((state, action, reward, next_state, done, time_passed))

    # @tf.function
    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return

    #     samples = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done, time_passed in samples:
    #         target = reward
    #         if not done:
    #             target = reward + (self.gamma ** time_passed) * tf.reduce_max(self.target_model(next_state[None, ...]))

    #         target_q_values = self.model(state[None, ...])
    #         indices = tf.constant([[0, action]])
    #         target_q_values = tf.tensor_scatter_nd_update(target_q_values, indices, tf.expand_dims(tf.cast(target, dtype=tf.float32), axis=0))

    #         with tf.GradientTape() as tape:
    #             predicted_q_values = self.model(state[None, ...])
    #             loss = self.loss_function(target_q_values, predicted_q_values)
    #         gradients = tape.gradient(loss, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones, time_passed_values = zip(*samples)

        states = tf.stack(states, axis=0)
        next_states = tf.stack(next_states, axis=0)
        time_passed_values = np.array(time_passed_values)
        discounts = self.gamma ** time_passed_values
        discounts[list(dones)] = 0

        target_rewards = np.array(rewards) + discounts * tf.reduce_max(self.target_model(next_states), axis=-1)
        target_q_values = self.model(states)
        target_indices = tf.stack([tf.range(batch_size), tf.convert_to_tensor(actions)], axis=-1)
        target_q_values = tf.tensor_scatter_nd_update(target_q_values, target_indices, target_rewards)

        with tf.GradientTape() as tape:
            predicted_q_values = self.model(states)
            loss = self.loss_function(target_q_values, predicted_q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def update_target_model(self):
        target_weights = self.target_model.trainable_variables
        model_weights = self.model.trainable_variables

        for target_w, model_w in zip(target_weights, model_weights):
            target_w.assign(self.tau * model_w + (1 - self.tau) * target_w)

# function for evaluating agent
def evaluate_agent(agent, env_eval):
    state = env_eval.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env_eval.step(action)
        total_reward += reward
        state = next_state

    print(f"Total market movement is {(env_eval.data[-1, 3] / env_eval.data[0, 3]).numpy()}")
    print(f"Evaluation Reward: {total_reward}, non-penalized Reward: {total_reward + env_eval.penalty * env_eval.total_steps}")
    return total_reward

if __name__ == "__main__":
    get_memory.get_memory_usage()

    # Initialize the training environment
    stock_window = 180
    max_trade_len = 500
    reward_penalty = 0.00001
    
    env = StockTradingEnvironment(duration=stock_window, max_trade_len=max_trade_len, penalty=reward_penalty)

    # Hyperparameters
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    print(f"State shape is {state_shape}")
    print(f"Action shape is {action_shape}")
    epsilon = 1.0
    epsilon_decay = 0.95
    risk_averse = 1.0
    risk_averse_decay = 0.95
    lr = 0.0001
    gamma = 0.9999
    soft_update = 0.9
    max_memory = 10000

    # Initialize the agent
    dqn_agent = DQNAgent(state_shape, action_shape, 
                         epsilon=epsilon, epsilon_decay=epsilon_decay, risk_averse=risk_averse,
                         learning_rate=lr, gamma=gamma, tau=soft_update, 
                         max_memory=max_memory)
    
    # Load the trained model weights
    if os.path.isfile('checkpoints/DQN_model.keras'):
        dqn_agent.model.load_weights('checkpoints/DQN_model.keras')
    dqn_agent.model.summary()
    # Count the trainable parameters
    print(sum(p.numel() for p in dqn_agent.model.trainable_variables))
    get_memory.get_memory_usage()

    # Initialize the evaluating environment
    env_eval = StockTradingEnvironment(duration=stock_window, max_trade_len=max_trade_len, penalty=reward_penalty)

    ticker = ["META"]
    eval_start_date = '2022-01-01'
    eval_end_date = '2022-12-31'
    env_eval.data = tf.convert_to_tensor(get_data(ticker[0], start_date=eval_start_date, end_date=eval_end_date), dtype=tf.float32)

    get_memory.get_memory_usage()

    # Training loop
    num_episodes = 100
    batch_size = 128
    agent_update_freq = 200    # number of steps to update agent
    eval_freq = 10

    evaluate_agent(dqn_agent, env_eval)

    for year in range(2000, 2022):
        start_date = str(year) + "-01-01"
        end_date = str(year) + "-12-31"

        env.data = tf.convert_to_tensor(get_data(ticker[0], start_date=start_date, end_date=end_date), dtype=tf.float32)
        if env.total_steps <= 0:
            continue

        print(f"Current env length is {env.total_steps}")
        print(f"Total market movement is {(env.data[-1, 3] / env.data[0, 3]).numpy()}")
        dqn_agent.clean_memory()

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                for _ in range(agent_update_freq):
                    action = dqn_agent.select_action(state)
                    next_state, reward, done, time_passed = env.step(action)
                    total_reward += reward
                    dqn_agent.remember(state, action, reward, next_state, done, time_passed)
                    state = next_state
                    if done:
                        break

                dqn_agent.replay(batch_size)
                dqn_agent.update_target_model()

            dqn_agent.epsilon *= dqn_agent.epsilon_decay
            if dqn_agent.epsilon < 0.1:
                dqn_agent.epsilon = 0.1
            dqn_agent.risk_averse *= risk_averse_decay
            if dqn_agent.risk_averse < 0.8:
                dqn_agent.risk_averse = 0.8

            print(f"Training Episode: {episode + 1}, Total Reward: {total_reward}, Total non-penalized Reward: {total_reward + env.penalty * env.total_steps}")
            if (episode + 1) % eval_freq == 0:
                evaluate_agent(dqn_agent, env_eval)

        # Save the trained model
        dqn_agent.model.save('checkpoints/DQN_model.keras')
