import random


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        if env in ['CartPole-v0', 'Pong-v0']:
            self.env = env
        else:
            raise ValueError("Unexpected environment")
        
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        
        self.eps = self.eps_start
        self.delta_eps = (self.eps_start - self.eps_end)/self.anneal_length

        self.init_nn()
        
    def init_nn(self):
        if self.env == 'CartPole-v0':
            self.init_nn_CartPole()
            self.forward = self.forward_CartPole
        elif self.env == 'Pong-v0':
            self.init_nn_Pong()
            self.forward = self.forward_Pong
        
    def init_nn_CartPole(self):
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def init_nn_Pong(self):
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_actions)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward_CartPole(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = x.squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward_Pong(self, x):
        #print('1',x.shape)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(self.relu(self.conv3(x)))
        #print('2',x.shape)
        #print(x)
        x = self.relu(self.fc1(x))
        #print(x)
        x = self.fc2(x)
        #print(x)
        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        # TODO: Implement epsilon-greedy exploration.
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        
        if self.eps > self.eps_end:
            self.eps -= self.delta_eps
        
        if not exploit and random.random() < self.eps:
            action = torch.tensor([random.randrange(self.n_actions)], device=device)
            
        else:
            action = torch.argmax(self.forward(observation))
            
        return action #self.mapAction(action)


        #     def mapAction(self, action):
        # if self.env == 'CartPole-v0':
        #     return action
        # elif self.env == 'Pong-v0':
        #     return action + 2
        

# In this function we have used code from the pytorch DQN tutorial
def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    obs, action, next_obs, reward = memory.sample(dqn.batch_size)
    non_final_mask = torch.tensor(tuple(map(lambda obs: obs is not None, next_obs)), device=device)
    #print(non_final_mask)
    non_final_next_obs = torch.cat([obs for obs in next_obs if obs is not None])
    #print(non_final_next_obs)
    obs = torch.cat(obs)
    action = torch.cat(action).reshape(-1, 1)
    reward = torch.cat(reward)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    #print(action)
    q_values = dqn.forward(obs).gather(1, action)
    
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    q_value_targets = torch.zeros(dqn.batch_size, device=device)
    q_value_targets[non_final_mask] = target_dqn.forward(non_final_next_obs).max(1)[0].detach()
    q_value_targets = dqn.gamma*q_value_targets + reward
    #print('detached')
    
    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
