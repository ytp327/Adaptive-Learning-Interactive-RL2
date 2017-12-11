from pacman import Directions
from game import Agent
import random
import util
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

class NormalQAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation", **args):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None
        self.prev_state = None
        self.prev_action = None
        if os.path.exists('Q.pkl'):
            with open('Q.pkl', 'rb') as f:
                self.Q = pickle.load(f)
            print('Lasted Q has been loaded.')
        else:
            self.Q = np.zeros([5, 5, 5, 5, 5, 2, 2, 4]) - 1000
            print('Cannot find Q backup, use default value')
        self.direction2num = {"West": 0, "East": 1, "North":2, "South": 3, "Stop":4}
        self.num2direction = ["West", "East", "North", "South", "Stop"]
        self.update_rate = 0.3
        self.explore_rate = 1
        self.count = 0
        self.score_list = []
        self.hundred_mean = []

    def sigint_handler(self):
        with open('Q.pkl', 'wb') as f:
            pickle.dump(self.Q, f)
        import sys
        sys.exit(0)

    def getAction(self, state):
        if self.prev_state is None:
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = [(state.generateSuccessor(0, action), action) for action in legal]
            scored = [(self.evaluationFunction(state), action) for state, action in successors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            self.prev_state = state
            self.prev_action = self.direction2num[random.choice(bestActions)]
            return self.num2direction[self.prev_action]

        reward = state.getScore() - self.prev_state.getScore()
        if reward == 1:
            reward = 2
        q_state = self.get_qstate(state)
        prev_qstate = self.get_qstate(self.prev_state)
        max_action = self.getGreedyAction(q_state, [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop'])
        max_qval = self.get_qval_ref(q_state)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + max_qval - prev_qval_ref[self.prev_action])
        # print(self.get_qval_ref(prev_qstate))
        self.prev_state = state
        self.prev_action = self.getGreedyAction(q_state, [self.direction2num[i] for i in self.prev_state.getLegalActions() if i != 'Stop'], self.explore_rate)
        return self.num2direction[self.prev_action]

    def final(self, state):
        qval = self.get_qval_ref(self.get_qstate(self.prev_state))
        if state.isWin():
            reward = 20
        else:
            reward = -500
        qval[self.prev_action] += self.update_rate * (reward - qval[self.prev_action])
        self.explore_rate += 0.2 / 100000
        if self.explore_rate > 1:
            self.explore_rate = 1
        # print(self.count, state.getScore())
        self.count+=1
        self.score_list.append(1 if state.isWin() else 0)
        if self.count % 100 == 0:
            self.hundred_mean.append(np.sum(self.score_list))
            print(self.count / 100, self.hundred_mean[-1])
            self.score_list = []
        if self.count >= 100000:
            plt.plot(list(range(0, len(self.hundred_mean))), self.hundred_mean)
            plt.show()
            self.sigint_handler()

    def getGreedyAction(self, q_state, legal, explore_rate=1):
        q_value = self.get_qval_ref(q_state)
        greedy_action = q_value.argmax()
        if greedy_action not in legal:
            greedy_action = legal[0]
        if len(legal) == 1:
            prob = [0] * 4
            prob[greedy_action] = 1
        else:
            prob = [(1 - explore_rate) / (len(legal)-1) if i in legal else 0 for i in range(4)]
            prob[greedy_action] = explore_rate
        return np.random.choice(4, p=prob)

    def get_qval_ref(self, q_state):
        q_value = self.Q
        for i in range(len(q_state)):
            q_value = q_value[q_state[i]]
        return q_value

    def get_qstate(self, state):
        q_state = [i-1 for i in state.getPacmanPosition()]
        q_state.extend([i-1 for i in state.getGhostPosition(1)])
        q_state.extend([self.direction2num[state.getGhostState(1).getDirection()]])
        q_state.append(1 if state.hasFood(3, 3) else 0)
        q_state.append(1 if state.hasFood(1, 1) else 0)
        q_state = [int(i) for i in q_state]
        return q_state

def scoreEvaluation(state):
    return state.getScore()

