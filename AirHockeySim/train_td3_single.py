#import gym
import numpy as np
import random
import math
from environment_single import AirHockeyEnvironment
import matplotlib.pyplot as plt
from matplotlib import style
import time
from datetime import datetime
from DDDQNAgent import TDAgent

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
            #print(env.paddle1.x_vel, env.paddle1.y_vel, action1)
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

            #with open('.\\data.csv', 'r') as r:
            #    lines = r.readlines()
            #avg_scores = [float(line.split(',')[2]) for line in lines][-200:]
            #plt.plot([j for j in range(len(avg_scores))], avg_scores, label="Avg scores")
            #plt.ylabel("Score")
            #plt.xlabel("Episode")
            #plt.legend()
            #time = datetime.now()
            #file_name = ".\\Plots\\Model_"+time.strftime("%d%m%Y")+"_"+str(i+1)+"_"+str(score1).split('.')[0]+".png"
            #plt.savefig(file_name, bbox_inches='tight')
            #plt.show(block=False)
            #plt.pause(5)
            #plt.close()

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

"""
MAYBE LEARNING RATE TOO HIGH
import gym
import numpy as np
import random
import math
from environment import AirHockeyEnvironment
from puckNET import DQAgent
import matplotlib.pyplot as plt
from matplotlib import style
import time
from datetime import datetime
from DDDQNAgent import Agent
if __name__ == "__main__":
    env = AirHockeyEnvironment()
    agent = DQAgent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=25, eps_end=0.01, input_dims=[8], lr=0.0003, eps_dec=1e-6)

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
                input_dims=[8], n_actions=25, mem_size=1000000, eps_min=0.01,
                batch_size=64, eps_dec=1e-3, replace=100)



    scores, eps_history, avg_scores = [], [], []
    n_games = 30000
    done = False
    for i in range(n_games):
        score = 0
        done = False
        paddle1_x, paddle1_x_vel, paddle1_y, paddle1_y_vel, puck_x, puck_x_vel, puck_y, puck_y_vel = env.reset()
        state = np.array([paddle1_x, paddle1_x_vel, paddle1_y, paddle1_y_vel, puck_x, puck_x_vel, puck_y, puck_y_vel])

        while not done:
            action1 = agent.choose_action(state)
            action2 = random.randint(0, 24)
            paddle1_x, paddle1_x_vel, paddle1_y, paddle1_y_vel, puck_x, puck_x_vel, puck_y, puck_y_vel, reward, done = env.step(action1, action2, render=True)
            new_state = np.array([paddle1_x, paddle1_x_vel, paddle1_y, paddle1_y_vel, puck_x, puck_x_vel, puck_y, puck_y_vel])
            score += reward
            agent.stored_transitions(state, action1, reward, new_state, done)
            agent.learn()
            state = new_state

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        latest_score = scores[-200:]
        latest_averages = avg_scores[-200:]
        print('episode ' + str(i) + '/' + str(n_games) +' score %.2f' % score, 'average score %.3f' % avg_score, 'eps %.3f' % eps_history[-1]) 
        if (i+1)%100 == 0 and i != 0:
            print("Saving Agent")
            agent.save_agent(i+1, score)
            plt.plot([i for i in range(min(200, len(latest_score)))], latest_score, label="Scores")
            plt.plot([i for i in range(min(200, len(latest_averages)))], latest_averages, label="Avg scores")
            plt.ylabel("Score")
            plt.xlabel("Episode")
            plt.legend()
            time = datetime.now()
            file_name = ".\\Plots\\Model_"+time.strftime("%d%m%Y")+"_"+str(i+1)+"_"+str(score).split('.')[0]+".png"
            plt.savefig(file_name, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(5)
            plt.close()

        plt.plot([i for i in range(len(avg_scores))], avg_scores, label="Avg scores")
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.legend()
        time = datetime.now()
        file_name = ".\\Plots\\latest_model_averages.png"
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    agent.save_agent(n_games, 999)
"""