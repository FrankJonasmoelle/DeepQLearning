import argparse

import matplotlib.pyplot as plt
import gym
from gym.wrappers import AtariPreprocessing
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize


from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=10, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}

def mapAction(action, env):
    if env == 'CartPole-v0':
        return action
    elif env == 'Pong-v0':
        return action + 2

def load_env(env_str):
    env = gym.make(env_str)
    if env_str == 'Pong-v0':
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    return env

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = load_env(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env=args.env, env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env=args.env, env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    ep=0

    # store return for plot
    returns = []
    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        next_obs_stack = torch.cat(env_config["obs_stack_size"] * [obs]).unsqueeze(0).to(device)
        
        t = 0
        while not done:
            obs_stack = next_obs_stack.clone()
            t += 1
            #print(obs_stack)
            # TODO: Get action from DQN.
            action = mapAction(dqn.act(obs_stack).item(), args.env)

            # Act in the true environment.
            next_obs, reward, done, info = env.step(action)

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
            else:
                next_obs_stack = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            memory.push(obs_stack, action, next_obs_stack, reward)

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
            if t % env_config["train_frequency"] == 0:
                optimize(dqn, target_dqn, memory, optimizer)

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if t % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)


            returns.append(mean_return)

            writer.add_scalar('return', mean_return,ep)
            ep=ep+1
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
        
    print(returns)
    plt.plot([10*x for x in range(len(returns))], returns)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average reward")
    plt.savefig(f"cart_pole_{env_config['target_update_frequency']}")
    plt.show()    
    # Close environment after training is completed.
    env.close()
    writer.flush()
    writer.close()
