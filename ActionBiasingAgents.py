from pacman import Directions
from game import Agent
import random
import util
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

TEACHER_Q_PATH = 'Q.pkl'
STUDENT_Q_PATH = 'agentQ2.pkl'

class ABAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation", **args):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None
        self.prev_state = None
        self.prev_action = None
        self.Q = self.create_qtable(TEACHER_Q_PATH)
        self.agent_Q = self.create_qtable(STUDENT_Q_PATH)
        self.B = np.zeros([5, 5, 5, 5, 5, 2, 2, 4]) + 1
        self.direction2num = {"West": 0, "East": 1, "North": 2, "South": 3, "Stop": 4}
        self.num2direction = ["West", "East", "North", "South", "Stop"]
        self.update_rate = 0.3
        self.explore_rate = 1
        self.count = 0
        self.score_list = []
        self.hundred_mean = []

    def create_qtable(self, dump_file=''):
        if not dump_file or not os.path.exists(dump_file):
            Q = np.zeros([5, 5, 5, 5, 5, 2, 2, 4]) - 1000
            print('Cannot find Q backup, use default value')
        else:
            with open(dump_file, 'rb') as f:
                Q = pickle.load(f)
            print('Lasted Q has been loaded.')
        return Q

    def sigint_handler(self):
        with open(STUDENT_Q_PATH, 'wb') as f:
            pickle.dump(self.agent_Q, f)
        import sys
        if self.count > 300000:
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
        q_state = self.get_qstate(state)
        prev_qstate = self.get_qstate(self.prev_state)
        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop'])
        max_qval = self.get_qval_ref(q_state, self.agent_Q)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + max_qval - prev_qval_ref[self.prev_action])
        # print(self.get_qval_ref(prev_qstate, self.agent_Q))
        self.prev_state = state
        self.prev_action = self.getGreedyAction(q_state,
                                                [self.direction2num[i] for i in self.prev_state.getLegalActions() if
                                                 i != 'Stop'], self.explore_rate)
        return self.num2direction[self.prev_action]

    def final(self, state):
        qval = self.get_qval_ref(self.get_qstate(self.prev_state), self.agent_Q)
        if state.isWin():
            reward = 500
        else:
            reward = -500
        qval[self.prev_action] += self.update_rate * (reward - qval[self.prev_action])
        self.explore_rate += 0.05 / 300000
        if self.explore_rate > 1:
            self.explore_rate = 1
        # print(self.count, state.getScore())
        self.count += 1
        self.score_list.append(1 if state.isWin() else 0)
        if self.count % 100 == 0:
            self.hundred_mean.append(np.sum(self.score_list))
            print(self.count / 100, self.hundred_mean[-1])
            self.score_list = []
            self.B -= 1. / 3000
        if self.count % 5000 == 0:
            plt.plot(list(range(0, len(self.hundred_mean))), self.hundred_mean)
            plt.show()
            self.sigint_handler()

    def getGreedyAction(self, q_state, legal, explore_rate=1):
        # just a copy of original self.Q
        q_value = self.get_qval_ref(q_state, self.agent_Q).copy()
        best_action = self.getTeacherGreedyAction(q_state)
        bias = self.action_bias(best_action)
        beliefs = self.get_bval_ref(q_state)
        q_value += beliefs * bias
        greedy_action = np.argmax(q_value)
        if greedy_action not in legal:
            greedy_action = legal[0]
        if len(legal) == 1:
            prob = [0] * 4
            prob[greedy_action] = 1
        else:
            prob = [(1 - explore_rate) / (len(legal) - 1) if i in legal else 0 for i in range(4)]
            prob[greedy_action] = explore_rate
        return np.random.choice(4, p=prob)

    def getTeacherGreedyAction(self, q_state):
        q_value = self.get_qval_ref(q_state, self.Q)
        greedy_action = q_value.argmax()
        return greedy_action

    def action_bias(self, best_action, C=0.8, rh=100):
        probs = [(1 - C) / 3] * 4
        probs[best_action] = C
        unconsist_best_action = np.random.choice(4, p=probs)
        bias = [-rh] * 4
        bias[unconsist_best_action] = rh
        return bias

    def get_qval_ref(self, q_state, Q):
        q_value = Q
        for i in range(len(q_state)):
            q_value = q_value[q_state[i]]
        return q_value

    def get_bval_ref(self, q_state):
        q_value = self.B
        for i in range(len(q_state)):
            q_value = q_value[q_state[i]]
        return q_value

    def get_qstate(self, state):
        q_state = [i - 1 for i in state.getPacmanPosition()]
        q_state.extend([i - 1 for i in state.getGhostPosition(1)])
        q_state.extend([self.direction2num[state.getGhostState(1).getDirection()]])
        q_state.append(1 if state.hasFood(3, 3) else 0)
        q_state.append(1 if state.hasFood(1, 1) else 0)
        q_state = [int(i) for i in q_state]
        return q_state

def scoreEvaluation(state):
    return state.getScore()
