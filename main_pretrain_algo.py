import numpy as np
import argparse
import torch
from copy import deepcopy
import os

from option_critic_pretrain import OptionCriticFeatures
from option_critic_pretrain import critic_loss as critic_loss_fn
from option_critic_pretrain import actor_loss as actor_loss_fn

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
parser.add_argument('--dual_gradient_descent', action='store_true', help='Whether to use dual gradient descent')

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
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
        load
        test(option_critic, args.env)
        return
    batch_size = args.batch_size
    lam = 0

    sum_entropy = 0
    task_dict = {}
    next_task_index = 0
    q_vals = np.zeros((12, 50, 6))
    for episode in range(1_00_00):
        print("episode: ", episode)
        options = []
        prev_step_termination = False
        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}
        obs, info = env.reset()
        state, task  = obs


        done = False ; truncated = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        if task in task_dict:
            current_option = task_dict[task]
        else:
            current_option = next_task_index
            task_dict[task] = next_task_index
            next_task_index += 1
        success, truncated = False, False
        steps = 0
        while ((not done) and (not truncated)):
            steps += 1
            import random
            if random.random() < 0.1:
                action = env.action_space.sample()  # Explore the action space
            else:
                action = np.argmax(q_vals[current_option, state])  # Exploit learned values

            # Apply the action and see what happens
            next_state, reward, done, truncated,  info = env.step(action)
            if reward == 20:
                print("done")
            next_state = next_state[0]

            current_value = q_vals[current_option, state, action]  # current Q-value for the state/action couple
            next_max = np.max(q_vals[current_option, next_state])  # next best Q-value

            # Compute the new Q-value with the Bellman equation
            q_vals[current_option, state, action] = (1 - 0.1) * current_value + 0.1 * (reward + 0.99 * next_max)
            state = next_state
        print(steps)
    print(q_vals)
    print(np.argmax(q_vals, axis = -1))
    breakpoint()
    return




def test(option_critic, env_name):
    # Note: there seems to be some bug in the test script as it does not match the training scripts performance
    # Perhaps the issue is that I'm taking argmax here instead of using temperature and epsilon
    #visualize_options(option_critic)
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

from tabulate import tabulate

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

    # Map each action in the policy to its corresponding label
    labeled_policy = [[action_labels[action] for action in row] for row in policy]

    # Create a table with borders for better readability
    table = tabulate(labeled_policy, tablefmt="fancy_grid")

    # Print the formatted table
    print(table)

import matplotlib.pyplot as plt
import numpy as np

from colorama import init, Fore, Back, Style
init(autoreset=True)

def fancy_color_coded_terminal_grid(numbers):
    # Initialize colorama
    init(autoreset=True)

    # Background colors for intensity
    colors = [
        Back.BLACK + Fore.WHITE,   # Very light for value 0
        Back.BLUE + Fore.WHITE,    # Light blue
        Back.CYAN + Fore.BLACK,    # Cyan
        Back.GREEN + Fore.BLACK,   # Green
        Back.YELLOW + Fore.BLACK,  # Yellow
        Back.LIGHTYELLOW_EX + Fore.BLACK, # Light yellow
        Back.LIGHTRED_EX + Fore.BLACK,    # Light red
        Back.RED + Fore.WHITE,    # Red
        Back.MAGENTA + Fore.WHITE,  # Magenta
        Back.LIGHTMAGENTA_EX + Fore.BLACK,  # Light magenta
        Back.WHITE + Fore.BLACK   # White for value 10
    ]

    # Print the grid with colors
    for row in numbers:
        row_str = ""
        for num in row:
            color = colors[min(int(num*10), 10)]  # Get the appropriate color
            row_str += color + f" {num:2} " + Style.RESET_ALL
        print(row_str)



#

def visualize_options(option_critic):
    for option in range(10):
        no_passenger = [[0 for _ in range(5)] for _ in range(5)]
        with_passenger = [[0 for _ in range(5)] for _ in range(5)]
        termination_probs_no_pass = [[0 for _ in range(5)] for _ in range(5)]
        termination_probs_with_pass = [[0 for _ in range(5)] for _ in range(5)]
        for taxi_state in range(25):
            with torch.no_grad():
                state = torch.zeros(26)
                state[taxi_state] = 1
                col = taxi_state % 5
                row = int((taxi_state - col)/5)
                no_passenger[row][col] = option_critic.get_greedy_action(state, option)
                termination_probs_no_pass[row][col] += option_critic.get_terminations(state)[option].item()
                state[-1] = 1
                with_passenger[row][col] = option_critic.get_greedy_action(state, option)
                termination_probs_with_pass[row][col] += option_critic.get_terminations(state)[option].item()
        print("OPTION:", option)
        print("no passenger:")
        pretty_print_policy(no_passenger)
        print("with passenger:")
        pretty_print_policy(with_passenger)
        print("no passenger:")
        print(termination_probs_no_pass)
        fancy_color_coded_terminal_grid(termination_probs_no_pass)
        print("with passenger:")
        fancy_color_coded_terminal_grid(termination_probs_with_pass)

#TODO: more intelligent heuristic







#TODO: temperature decay

#TODO: Do some kind of testing where you check if the learned policies are optimal, and also print out the low level policies from each state

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
