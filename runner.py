import numpy as np
from itertools import count
import torch
import rl_agents
import models
import statistics
from tqdm import tqdm
import os
import time
from statistics import mean
import random

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

class Runner:
    ''' run an agent in an environment '''
    def __init__(self, train_env, test_env, agent, training):
        self.train_env = train_env
        self.test_env = test_env # environment for testing
        self.agent = agent
        self.training = training

    def play_game(self, num_iterations, epsilon, training=True, time_usage=False, one_time=False):
        if training:
            self.env = self.train_env
        else:
            self.env = self.test_env
        if time_usage:
            total_time = 0.0 # total time for all iterations on all testing graphs

        for iteration in range(num_iterations):

            if training:
                for g_idx in range(len(self.env.graphs)):
                    self.env.reset(g_idx)
                    for i in range(self.env.budget):
                        if epsilon>=random.uniform(0,1):
                            actions=random.sample(self.env.action,1)[0]
                        else:
                            actions = self.agent.select_action(self.env.graph_realization,self.env.graph, self.env.actions,budget=self.env.budget)
                        self.env.step(actions)
                        self.agent.memorize(self.env)
            else:
                c_rewards = []
                im_seeds = []
                for g_idx in range(len(self.env.graphs)):
                    if time_usage:
                        start_time = time.time()
                    seeds=[]
                    self.env.reset(g_idx, training=training)
                    for i in range(self.env.budget):
                        action = self.agent.select_action(self.env.graph_realization,self.env.graph,self.env.actions,budget=self.env.budget)
                        self.env.step(action)
                    # no sort of actions selected
                        seeds.append(action)

                    if time_usage:
                        total_time += time.time() - start_time
                    final_reward=[]
                    self.env.reset(g_idx, training=training)
                    for action in seeds:
                        r1,_=self.env.compute_reward_realization(action)
                        final_reward.append(r1)
                    c_rewards.append(final_reward)
                    im_seeds.append(seeds)
                    if time_usage:
                        print(f'Seed set generation per iteration time usage is: {total_time/num_iterations:.2f} seconds')
                return c_rewards, im_seeds


    def train(self, num_epoch, model_file, result_file):
        ''' let agent act and learn from the environment '''
        # pretrain
        tqdm.write('Pretraining:')
        self.play_game(1000, 1.0)

        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0
        # train
        tqdm.write('Starting fitting:')
        progress_fitting = tqdm(total=num_epoch)
        for epoch in range(num_epoch):
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - epoch) / eps_step)
            
            if epoch % 10 == 0:
                self.play_game(10, eps)

            if epoch % 10 == 0 and epoch!=0:
                # test
                rewards, seeds = self.play_game(1, 0.0, training=False)
                tqdm.write(f'{epoch}/{num_epoch}: ({str(seeds)}) | {rewards}')                

            if epoch % 1000 == 0:
                # save model
                self.agent.save_model(model_file + str(epoch))
            if epoch % 20 == 0:
                self.agent.update_target_net()
            # train the model
            self.agent.fit()
            progress_fitting.update(1)
        # show test results after training
        rewards, seeds = self.play_game(1, 0.0, training=False)
        tqdm.write(f'{num_epoch}/{num_epoch}: ({str(seeds)[1:-1]}) | {rewards}')

        self.agent.save_model(model_file)


    def test(self, num_trials=1):
        ''' let agent act in the environment
            num_trials: may need multiple trials to get average
        '''
        print('Generate seeds at one time:', flush=True)
        all_rewards, all_seeds = self.play_game(num_trials, 0.0, False, time_usage = True, one_time = True)
        print(all_rewards,all_seeds)
