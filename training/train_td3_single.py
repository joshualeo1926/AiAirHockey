#import gym
import numpy as np
import random
import math
from environment_single import AirHockeyEnvironment
import matplotlib.pyplot as plt
from matplotlib import style
import time
from datetime import datetime
from TD3Agent import TDAgent

if __name__ == "__main__":
    env = AirHockeyEnvironment()
    n_games = 20000
    load_checkpoint = True

    agent1 = TDAgent(alpha=5e-5, beta=5e-5, input_dims=[8], tau=0.005, max_action=1.0, 
                    min_action=-1.0, gamma=0.99, update_actor_interval=20, 
                    warmup=0, n_actions=2, max_size=1000000, 
                    layer1_size=400, layer2_size=300, batch_size=100,
                    noise=0.1)

    agent2 = TDAgent(alpha=5e-5, beta=5e-5, input_dims=[8], tau=0.005, max_action=1.0, 
                    min_action=-1.0, gamma=0.99, update_actor_interval=20, 
                    warmup=0, n_actions=2, max_size=1000000, 
                    layer1_size=400, layer2_size=300, batch_size=100,
                    noise=0.1)      

    if load_checkpoint:
        agent1.load_models()
        agent2.load_models()
    
    epsilon = 0
    epsilon2 = 0
    for i in range(n_games):
        done = False
        if np.random.random() < epsilon:
            agent1.warmup = 250000
        else:
            agent1.warmup = 0
        state1, state2 = env.reset()
        state1 = np.array(state1)
        state2 = np.array(state2)
        score1 = 0
        score2 = 0
        while not done:
            if np.random.random() < epsilon2:
                agent1.warmup = 250000
            else:
                agent1.warmup = 0
            action1 = agent1.choose_actions(state1)
            action2 = agent1.choose_actions(state2)
            new_state1, new_state2, reward1, reward2, done = env.step(action1, action2, render=True)
            new_state1 = np.array(new_state1)
            new_state2 = np.array(new_state2)

            score1 += reward1
            score2 += reward2

            agent1.remember(state1, action1, reward1, new_state1, done)
            agent1.learn()

            state1 = new_state1
            state2 = new_state2

        with open('.\\data.csv', 'r') as r:
                lines = r.readlines()

        scores = []
        avg_score = 0
        if len(lines) >= 100:
            for k in range(100):
                scores.append(float(lines[(-100+k)].split(',')[0]))
            avg_score = np.mean(scores)

        elif len(lines) > 0:
            for line in lines:
                scores.append(float(line.split(',')[0]))
            avg_score = np.mean(scores)
        
        if i == 0:
            avg_score = score1

        with open('.\\data.csv', 'a') as a:
            a.write('%.2f' % score1 + ',' + '%.3f' % 0.0000 + ',' + '%.3f' % avg_score + '\n')

        print('episode ' + str(i) + '/' + str(n_games) +' score %.2f' % score1, 'average score %.3f' % avg_score) 
        if (i+1)%100 == 0 and i != 0:

            agent1.save_models()

        with open('.\\data.csv', 'r') as r:
            lines = r.readlines()
        avg_scores = [float(line.split(',')[2]) for line in lines]
        plt.plot([j for j in range(len(avg_scores))], avg_scores, label="Avg scores")
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.legend()
        time = datetime.now()
        file_name = ".\\Plots\\latest_model_averages.png"
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()