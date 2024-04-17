import numpy as np
import argparse
import torch
from copy import deepcopy
import os

from option_critic import OptionCriticFeatures, OptionCriticConv, deoc_entropy
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import Logger

import time

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=200, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=2, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')
parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(4e6), help='number of maximum steps to take.') # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default="test", help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')
parser.add_argument('--model', type=str, default=None, help='use pretrained model')
parser.add_argument('--test', type=int, default=0, help='only do testing, make sure to also pass model arg')
parser.add_argument('--diversity_learning', action='store_true', help='Whether to use diversity enriched learning')
parser.add_argument('--diversity_termination', action='store_true', help='Whether to use diversity enriched termination')
parser.add_argument('--diversity_tradeoff', type=float, default=0.0001, help='Tradeoff between diversity and reward')
parser.add_argument('--deoc_entropy_samples', type=int, default=6, help='Number of samples to estimate entropy')
parser.add_argument('--separate_value_function', action='store_true', help='Whether to use separate termination network')

def save_model_with_args(model, run_name, arg_string, ep_num):
    # Create the directory path
    run_name += f"-{ep_num}"

    model_dir = os.path.join('models', run_name)
    # Create directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)
    # Define the path for saving the model and arguments
    model_path = os.path.join(model_dir, 'model.pth')
    args_path = os.path.join(model_dir, 'args')
    # Save the model state
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully at {model_path}")
    # Write the argument string to the args file
    with open(args_path, 'w') as file:
        file.write(arg_string)
    print(f"Arguments saved successfully at {args_path}")


def load_model(model, run_name):
    # Define the directory path from which to load the model and arguments
    model_dir = os.path.join('models', run_name)
    model_path = os.path.join(model_dir, 'model.pth')
    # Load the model state
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_path}")
    else:
        raise FileNotFoundError(f"No model file found at {model_path}")

def run(args):
    env, is_atari = make_env(args.env, render_mode = None)
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
    if args.model:
        print("Loading model...")
        load_model(option_critic, args.model)
    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    run_name = f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{args.seed}-{int(time.time())}"
    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=run_name)

    steps = 0 ;
    if args.switch_goal: print(f"Current goal {env.goal}")
    if args.test==1:
        test(option_critic, args.env)
        return
    batch_size = args.batch_size
    lam = 0

    sum_entropy = 0

    for episode in range(10_000):
        options = []
        prev_step_termination = False
        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        obs, info   = env.reset()
        full_obs, local_obs = obs
        full_state, local_state = option_critic.get_state(to_tensor(full_obs)), option_critic.get_state(to_tensor(local_obs))
        greedy_option  = option_critic.greedy_option(full_state)
        current_option = 0

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).
        if args.switch_goal and logger.n_eps == 1000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={args.seed}_1k')
            env.switch_goal()
            print(f"New goal {env.goal}")

        if args.switch_goal and logger.n_eps > 2000:
            torch.save({'model_params': option_critic.state_dict(),
                        'goal_state': env.goal},
                        f'models/option_critic_seed={args.seed}_2k')
            break

        done = False ; truncated = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0

        switch_loss = 0
        success = False
        while ((not done) and (not truncated)) and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
            options.append(current_option)
    
            action, logp, entropy = option_critic.get_action(local_state, current_option)

            next_obs, reward, done, truncated, info = env.step(action)

            if args.diversity_learning:
                entropy_loss = deoc_entropy(option_critic, local_state, option_critic.options_W, args)
                sum_entropy += entropy_loss
                pseudo_reward = (1 - args.diversity_tradeoff) * reward + args.diversity_tradeoff * entropy_loss
                reward = pseudo_reward

            n_full_obs, n_local_obs = next_obs
            buffer.push(obs, current_option, reward, next_obs, done, action)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > batch_size: # after first few iters this is satisfied every time!
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, args, sum_entropy / steps)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    loss += critic_loss
                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            local_state = option_critic.get_state(to_tensor(next_obs[1]))
            full_state = option_critic.get_state(to_tensor(next_obs[0]))
            option_termination, greedy_option, termination_prob = option_critic.predict_option_termination(full_state, local_state, current_option)
            switch_loss += termination_prob
            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            # TODO - add model saving
            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        if episode % 500 == 0:
            save_model_with_args(option_critic, run_name, str(args), episode)
        # Uncomment this to try increasing option size with dual gradient descent
        # if success:
        #     lam += 7e-5
        # else:
        #     lam -= 2e-5
        # if lam < 0:
        #     lam = 0
        # loss = lam * switch_loss
        # optim.zero_grad()
        # loss.backward()
        # optim.step()
        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        print(lam)

    save_model_with_args(option_critic, run_name, str(args))
    test(option_critic, args.env)

def test(option_critic, env_name):
    # Note: there seems to be some bug in the test script as it does not match the training scripts performance
    # Perhaps the issue is that I'm taking argmax here instead of using temperature and epsilon
    visualize_options(option_critic)
    option_critic.testing = True
    option_critic.temperature = 0.01 #TODO
    env, is_atari = make_env(env_name, render_mode="human")
    input("press enter to see visualizations")
    for i in range(5):
        obs, info = env.reset()
        full_obs, local_obs = obs
        full_state, local_state = option_critic.get_state(to_tensor(full_obs)), option_critic.get_state(
            to_tensor(local_obs))
        greedy_option = option_critic.greedy_option(full_state)
        option_termination = True
        done, truncated = False, False
        actions = []
        options = []
        steps = 0
        while ((not done) and (not truncated)) and steps < 30:
            steps += 1
            time.sleep(0.5)
            if option_termination:
                current_option =  greedy_option
            options.append(current_option)
            action, logp, entropy = option_critic.get_action(local_state, current_option)
            actions.append(action)
            next_obs, reward, done, truncated, info = env.step(action)
            local_state = option_critic.get_state(to_tensor(next_obs[1]))
            full_state = option_critic.get_state(to_tensor(next_obs[0]))
            option_termination, greedy_option, prob = option_critic.predict_option_termination(full_state, local_state,
                                                                                         current_option)
        print("options: ", options)
        print("actions: ", actions)

def pretty_print_policy(policy):
    # Define action labels according to the policy description
    action_labels = {
        0: "↓",   # Move south (down)
        1: "↑",   # Move north (up)
        2: "→",   # Move east (right)
        3: "←",   # Move west (left)
        4: "P",   # Pickup passenger
        5: "D"    # Drop off passenger
    }

    # Iterate through each row in the policy
    for row in policy:
        # Map each action in the row to its corresponding label
        labeled_row = [action_labels[action] for action in row]
        # Join the labeled actions with spaces for better readability and print
        print(' '.join(labeled_row))

def visualize_options(option_critic):
    for option in range(10):
        no_passenger = [[0 for _ in range(5)] for _ in range(5)]
        with_passenger = [[0 for _ in range(5)] for _ in range(5)]
        for taxi_state in range(24):
            with torch.no_grad():
                state = torch.zeros(26)
                state[taxi_state] = 1
                col = taxi_state % 5
                row = int((taxi_state - col)/5)
                no_passenger[row][col] = option_critic.get_greedy_action(state, option)
                state[-1] = 1
                with_passenger[row][col] = option_critic.get_greedy_action(state, option)
        print("OPTION:", option)
        print("no passenger:", pretty_print_policy(no_passenger))
        print("with passenger:", pretty_print_policy(with_passenger))
#TODO: more intelligent heuristic







#TODO: temperature decay

#TODO: Do some kind of testing where you check if the learned policies are optimal, and also print out the low level policies from each state

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
