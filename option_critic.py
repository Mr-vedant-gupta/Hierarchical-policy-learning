import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor


class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        # self.num_options = num_options
        # self.magic_number = 7 * 7 * 64
        # self.device = device
        # self.testing = testing
        #
        # self.temperature = temperature
        # self.eps_min   = eps_min
        # self.eps_start = eps_start
        # self.eps_decay = eps_decay
        # self.eps_test  = eps_test
        # self.num_steps = 0
        #
        # self.features = nn.Sequential(
        #     nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.modules.Flatten(),
        #     nn.Linear(self.magic_number, 512),
        #     nn.ReLU()
        # )
        #
        # self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        # self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        # self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        # self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        #
        # self.to(device)
        # self.train(not testing)

    # def get_state(self, obs):
    #     if obs.ndim < 4:
    #         obs = obs.unsqueeze(0)
    #     obs = obs.to(self.device)
    #     state = self.features(obs)
    #     return state
    #
    # def get_Q(self, state):
    #     return self.Q(state)
    #
    # def predict_option_termination(self, state, current_option):
    #     termination = self.terminations(state)[:, current_option].sigmoid()
    #     option_termination = Bernoulli(termination).sample()
    #
    #     Q = self.get_Q(state)
    #     next_option = Q.argmax(dim=-1)
    #     return bool(option_termination.item()), next_option.item()
    #
    # def get_terminations(self, state):
    #     return self.terminations(state).sigmoid()
    #
    # def get_action(self, state, option):
    #     logits = state.data @ self.options_W[option] + self.options_b[option]
    #     action_dist = (logits / self.temperature).softmax(dim=-1)
    #     action_dist = Categorical(action_dist)
    #
    #     action = action_dist.sample()
    #     logp = action_dist.log_prob(action)
    #     entropy = action_dist.entropy()
    #
    #     return action.item(), logp, entropy
    #
    # def greedy_option(self, state):
    #     Q = self.get_Q(state)
    #     return Q.argmax(dim=-1).item()
    #
    # @property
    # def epsilon(self):
    #     if not self.testing:
    #         eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
    #         self.num_steps += 1
    #     else:
    #         eps = self.eps_test
    #     return eps


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        # self.features = nn.Sequential(
        #     nn.Linear(in_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU()
        # )

        self.Q            = nn.Linear(500, num_options, bias = False)                 # Policy-Over-Options
        self.terminations = nn.Linear(26, num_options, bias = False)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 26, num_actions))
        self.Q_opt = nn.Linear(500, num_actions, bias = False)
        #self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = obs #self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, full_state, local_state, current_option):
        termination = self.terminations(local_state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(full_state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item(), termination
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_option_pmf(self, state, option):
        logits = state.data @ self.options_W[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        return action_dist

    def get_action(self, state, option):
        action_dist = self.get_option_pmf(state, option)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

    def get_greedy_action(self, state, option):
        logits = state.data @ self.options_W[option] #+ self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = torch.argmax(action_dist.probs)
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item()
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps

def deoc_entropy(model, obs, option_policies, args):
    import itertools
    cum_entropy = 0
    # print(f"len(option_policies): {len(option_policies)}")
    num_samples = min(len(option_policies), args.deoc_entropy_samples)
    combinations = list(itertools.combinations(range(len(option_policies)),2))
    for _ in range(num_samples):
        sample = combinations[np.random.randint(0,len(combinations))]
        sampled_op1 = sample[0]
        sampled_op2 = sample[1]
        # print(sampled_op1)
        # print(sampled_op2)
        x1 = model.get_option_pmf(obs, sampled_op1)
        x2 = model.get_option_pmf(obs, sampled_op2)
        # print(x1, x2)
        x1 = torch.clip(x1,1e-20, 1.0)
        x2 = torch.clip(x2,1e-20, 1.0)
        cum_entropy += -torch.sum(x1*torch.log(x2))/x1.shape[0]
    return cum_entropy/(num_samples)

def critic_loss(model, model_prime, data_batch, args):
    full_obs, local_obs, options, rewards, nfull_obs, nlocal_obs, dones, actions = data_batch
    # full_obs, local_obs = np.array([o[0] for o in obs]), np.array([o[1] for o in obs])
    # nfull_obs, nlocal_obs = np.array([o[0] for o in next_obs]), np.array([o[1] for o in next_obs])
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options).to(model.device)
    actions = torch.LongTensor(actions).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    full_states = model.get_state(to_tensor(full_obs)).squeeze(0)
    local_states = model.get_state(to_tensor(local_obs)).squeeze(0)
    Q      = model.get_Q(full_states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    nfull_states_prime = model_prime.get_state(to_tensor(nfull_obs)).squeeze(0)
    nlocal_states_prime = model_prime.get_state(to_tensor(nlocal_obs)).squeeze(0)
    next_Q_prime      = model_prime.get_Q(nfull_states_prime) # detach?
    # Additionally, we need the beta probabilities of the next state
    nfull_states            = model.get_state(to_tensor(nfull_obs)).squeeze(0)
    nlocal_states = model.get_state(to_tensor(nlocal_obs)).squeeze(0)
    next_termination_probs = model.get_terminations(nlocal_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * (-args.termination_reg + next_Q_prime.max(dim=-1)[0])) #TODO: will it help to add terminatrion reg here too?

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    #breakpoint()
    q_error = (model.Q_opt(full_states)[batch_idx, 0] - (rewards + masks*args.gamma*model_prime.Q_opt(nfull_states_prime).max(dim=-1)[0]).detach()).pow(2).mul(0.5).mean()
    return td_err + q_error

def actor_loss(obs, option, logp, entropy, reward, done, next_obs, model, model_prime, args, avg_entropy=0.):
    print(args.termination_reg)
    full_obs, local_obs = obs
    nfull_obs, nlocal_obs = next_obs
    #TODO make sure the line below is now reachable
    # if local_obs[-1] == 1:
    #     breakpoint()
    full_state = model.get_state(to_tensor(full_obs))
    local_state = model.get_state(to_tensor(local_obs))
    nfull_state = model.get_state(to_tensor(nfull_obs))
    nlocal_state = model.get_state(to_tensor(nlocal_obs))
    nfull_state_prime = model_prime.get_state(to_tensor(nfull_obs))

    next_option_term_prob = model.get_terminations(nlocal_state)[:, option].detach()

    Q = model.get_Q(full_state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(nfull_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
         ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    # Switch termination loss stattement here to try using separate value funciton
    # termination_loss = option_term_prob * (Q[option].detach() - model.Q_opt(full_state).detach().squeeze().max(dim=-1)[0].detach()) * (
    #             1 - done)

    if args.diversity_termination:
        deoc_ent = deoc_entropy(model, local_state, model.options_W, args) - avg_entropy
        termination_loss = deoc_ent
    elif args.separate_value_function:
        option_term_prob = model.get_terminations(local_state)[:, option]
        termination_loss = option_term_prob * (Q[option].detach() - model.Q_opt(full_state).detach().squeeze().max(dim=-1)[0].detach()) * (1 - done)
    else:
        option_term_prob = model.get_terminations(local_state)[:, option]
        if args.new_termination:
            if option == Q.argmax(dim=-1).item():
                sorted_tensor, _ = Q.sort(dim=-1, descending=True)
                second_largest = sorted_tensor[..., 1].detach()
                termination_loss = option_term_prob * (
                    Q[option].detach() - second_largest + args.termination_reg) * (1 - done)
            else:
                termination_loss = option_term_prob * (
                    Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)
        else:
            termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg) * (1 - done)

    # print(termination_loss)

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss
