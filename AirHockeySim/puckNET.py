import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from datetime import datetime

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0, lr=0.0001):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(T.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        self.device = T.device('cpu')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.squeeze().exp().expand_as(mu)
        dist = T.distributions.Normal(mu, std)
        return dist, value


class Agent():
    def __init__(self, hidden_size=64, lr=1e-4, gamma=0.99, tau=0.95,
                epsilon=0.2, cirtic_discount=0.5, entropy_beta=0.001, 
                mini_batch_size=64, ppo_steps=256, ppo_epochs=10,
                num_inputs=4, num_outputs=2):
        self.ActorCritic = ActorCritic(num_inputs=num_inputs, num_outputs=num_outputs, hidden_size=hidden_size, std=0.0, lr=lr)
        self.entropy = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.cirtic_discount = cirtic_discount
        self.entropy_beta = entropy_beta
        self.mini_batch_size = mini_batch_size
        self.ppo_steps = ppo_steps
        self.ppo_epochs = ppo_epochs
        self.num_outputs = num_outputs

        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def choose_action(self, state):
        dist, value = self.ActorCritic.forward(T.tensor(state, dtype=T.float).to(self.ActorCritic.device))
        action = dist.sample()

        return action, dist, value

    def store_episode(self, action, dist, value, reward, done, state):
        log_prob = dist.log_prob(action).to(self.ActorCritic.device)
        self.log_probs.append(log_prob.detach().numpy()) 
        self.values.append(value)
        self.rewards.append(T.tensor(reward, dtype=T.float).unsqueeze(-1).to(self.ActorCritic.device))
        self.masks.append(T.tensor(1-done, dtype=T.float).unsqueeze(-1).to(self.ActorCritic.device))
        self.states.append(state)
        self.actions.append(action.cpu().numpy())

    def clear_memory(self):
        del self.log_probs[:]
        del self.values[:]
        del self.rewards[:]
        del self.masks[:]
        del self.states[:]
        del self.actions[:]

    def compute_gae(self, next_value):
        tempvalues = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * tempvalues[step + 1] * self.masks[step] - tempvalues[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            returns.insert(0, gae + tempvalues[step])
        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            #print(rand_ids)
            #print(states[rand_ids])
            #print(actions[rand_ids])
            #print("logprob", log_probs[rand_ids])
            #print(returns[rand_ids])
            #print(advantage[rand_ids])
            if self.num_outputs != 1 and False:
                yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
            else:
                yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]


    def ppo_update(self, returns):
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        returns = T.cat(returns).detach()
        log_probs = T.tensor(self.log_probs, dtype=T.float).to(self.ActorCritic.device)
        values = T.cat(self.values).detach()
        #states = T.cat(self.states)
        states = T.tensor(self.states, dtype=T.float).to(self.ActorCritic.device)
        #actions = T.cat(self.actions)
        actions = T.tensor(self.actions, dtype=T.float).to(self.ActorCritic.device)
        advantages = returns - values
        advantages = normalize(advantages)

        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                
                dist, value = self.ActorCritic.forward(T.FloatTensor(state).to(self.ActorCritic.device))
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action).to(self.ActorCritic.device)
                print("new_log_probs", new_log_probs)
                #print(old_log_probs)
                ratio = (new_log_probs - old_log_probs).exp()
                print("ratio", ratio)
                print("advantage", advantage)
                surr1 = ratio * advantage
                surr2 = T.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage

                actor_loss  = - T.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.cirtic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                self.ActorCritic.optimizer.zero_grad()
                loss.backward()
                self.ActorCritic.optimizer.step()

                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy
                
                count_steps += 1


                #print(
                #    "===========",
                #    "\nReturns: %.2f" % (sum_returns / count_steps), 
                #    "\nAdvantages: %.2f" % (sum_advantage / count_steps),  
                #    "\nActor Loss: %.2f" % (sum_loss_actor / count_steps), 
                #    "\nCritic Loss: %.2f" % (sum_loss_critic / count_steps),
                #    "\nTotal Loss: %.2f" % (sum_loss_total / count_steps),
                #    "\nEntropy: %.2f" % (sum_entropy / count_steps), 
                #    "\n===========\n",
                #    )


    def save_agent(self, itter, score):
        time = datetime.now()
        file_name = "Model_"+time.strftime("%d%m%Y")+"_"+itter+"_"+str(score)+".dat"
        T.save(self.ActorCritic.state_dict(), ".\\DeepQModels\\"+file_name)


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)
        self.fc4 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu') #'cuda:0' if T.cuda.is_available() else 'cpu:0'
        self.to(self.device)

    def forward(self, state):
        #state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions

class DQAgent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=1000000, eps_end=0.05, eps_dec=1e-5):
        """
        gamma = discount factor it tells the agent how much to discount fututre rewards
        epsilon = epsilon greedy
            makes it mainly take random actions most of the time then best action some of the time, then the random decreases over time
        eps_end = end epsilon
        eps_dec = how much to decrement epsilon by
        eps_min = never want epsilon to go to zero becuase you never know if ur estimate is accurate
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions,
                                  input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), 
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory =  np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory =  np.zeros(self.mem_size, dtype=np.bool)

    def stored_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state.float())
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)                           #target network
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
            
    def save_agent(self, itter, score):
        time = datetime.now()
        file_name = "Model_"+time.strftime("%d%m%Y")+"_"+str(itter)+"_"+str(score).split('.')[0]+".dat"
        T.save(self.Q_eval.state_dict(), ".\\DeepQModels\\"+file_name)
