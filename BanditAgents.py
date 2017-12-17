from pacman import Directions
from game import Agent
import random
import util
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

TEACHER_Q_PATH = 'NormalQ.pkl'
STUDENT_Q_PATH = 'agentQBA.pkl'

class MBAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation", **args):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None
        self.prev_state = None
        self.prev_action = None
        self.Q = self.create_qtable(TEACHER_Q_PATH)
        self.agent_Q = self.create_qtable(STUDENT_Q_PATH)
        self.B = 1
        self.look_prob = 1
        self.direction2num = {"West": 0, "East": 1, "North": 2, "South": 3, "Stop": 4}
        self.num2direction = ["West", "East", "North", "South", "Stop"]
        self.update_rate = 0.1
        self.explore_rate = 0.9
        self.count = 0
        self.score_list = []
        self.hundred_mean = []
        self.need_change = True
        self.total_episode = 10000
        self.L = 0.1
        self.sim = np.ones([4])
        self.methods = [self.get_action_action_bias,
                        self.get_action_control_sharing,
                        self.get_action_reward_shaping,
                        self.get_action_q_update]
        self.action_bias_position = 0
        self.cur_method = None
        self.decay = 0.7
        self.weights = np.ones([4])

    def get_softmax_prob(self, who):
        who_prob = np.exp(0.3*(self.weights[who] - self.weights.min()))
        all_prob = sum([np.exp(0.3*(bandit - self.weights.min())) for bandit in self.weights])
        return who_prob / all_prob

    def sample_softmax(self):
        probs = [self.get_softmax_prob(x) for x in range(4)]
        return np.random.choice(4, p=probs)

    def get_normalized_reward(self, reward):
        reward = reward + 500/(1-self.decay)
        reward /= 1000/(1-self.decay)
        return reward

    def get_method_prob(self, state, explore_rate=1., action_bias=False):
        # just a copy of original self.Q
        q_state = self.get_qstate(state)
        q_value = self.get_qval_ref(q_state, self.agent_Q).copy()
        # beliefs = self.get_bval_ref(q_state)
        if action_bias:
            best_action = self.getTeacherGreedyAction(q_state)
            bias = self.action_bias(best_action)
            q_value += self.B * bias
        greedy_action = np.argmax(q_value)
        prob = [(1 - explore_rate) / 3 for i in range(4)]
        prob[greedy_action] = explore_rate
        return prob

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
        if self.need_change or self.cur_method is None:
            self.cur_method = self.methods[self.sample_softmax()]
            # print(' '.join(self.cur_method.__name__.split('_')[2:]))
            self.need_change = False

        action = self.cur_method(state)
        for x in range(4):
            if x == self.action_bias_position:
                self.sim[x] *= self.get_method_prob(state, self.explore_rate, action_bias=True)[action]
            else:
                self.sim[x] *= self.get_method_prob(state, self.explore_rate)[action]

        return self.num2direction[action]

    def update_multi_bandit(self, reward):
        # reward: environment's and human's reward
        for x in range(4):
            self.weights[x] += 0.1 * self.sim[x]*(self.get_normalized_reward(reward) - self.weights[x])/self.sim.sum()
        self.sim = np.ones([4])

    def get_action_action_bias(self, state):
        if self.prev_state is None:
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = [(state.generateSuccessor(0, action), action) for action in legal]
            scored = [(self.evaluationFunction(state), action) for state, action in successors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            self.prev_state = state
            self.prev_action = self.direction2num[random.choice(bestActions)]
            return self.prev_action

        reward = state.getScore() - self.prev_state.getScore()
        q_state = self.get_qstate(state)
        prev_qstate = self.get_qstate(self.prev_state)
        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop'], True)
        max_qval = self.get_qval_ref(q_state, self.agent_Q)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + self.decay * max_qval - prev_qval_ref[self.prev_action])

        # print(self.get_qval_ref(prev_qstate, self.agent_Q))
        self.prev_state = state
        self.prev_action = self.getGreedyAction(q_state,
                                                [self.direction2num[i] for i in self.prev_state.getLegalActions() if
                                                 i != 'Stop'], self.explore_rate)
        return self.prev_action

    def get_action_reward_shaping(self, state):
        if self.prev_state is None:
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = [(state.generateSuccessor(0, action), action) for action in legal]
            scored = [(self.evaluationFunction(state), action) for state, action in successors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            self.prev_state = state
            self.prev_action = self.direction2num[random.choice(bestActions)]
            return self.prev_action

        reward = state.getScore() - self.prev_state.getScore()
        prev_qstate = self.get_qstate(self.prev_state)
        reward += self.B * self.reward_bias(self.prev_action, self.getTeacherGreedyAction(prev_qstate))
        # print(reward)
        q_state = self.get_qstate(state)
        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop'])
        max_qval = self.get_qval_ref(q_state, self.agent_Q)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + self.decay * max_qval - prev_qval_ref[self.prev_action])
        # print(self.get_qval_ref(prev_qstate, self.agent_Q))
        # print(self.get_qval_ref(prev_qstate, self.Q))
        # print(self.prev_action)
        # print('-------------------------------------------')
        self.prev_state = state
        self.prev_action = self.getGreedyAction(q_state,
                                                [self.direction2num[i] for i in self.prev_state.getLegalActions() if
                                                 i != 'Stop'], self.explore_rate)
        return self.prev_action

    def get_action_control_sharing(self, state):
        if self.prev_state is None:
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = [(state.generateSuccessor(0, action), action) for action in legal]
            scored = [(self.evaluationFunction(state), action) for state, action in successors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            self.prev_state = state
            self.prev_action = self.direction2num[random.choice(bestActions)]
            return self.prev_action

        prev_qstate = self.get_qstate(self.prev_state)
        q_state = self.get_qstate(state)
        control_prob = random.random()
        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if
                                           i != 'Stop'])

        if control_prob < self.look_prob:
            best_action = self.getTeacherGreedyAction(q_state)
            current_action = self.getUnConsistentAction(best_action)
            legal = [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop']
            if current_action not in legal:
                current_action = self.direction2num[[i for i in state.getLegalActions() if i != 'Stop'][0]]
        else:
            current_action = self.getGreedyAction(q_state,
                                                    [self.direction2num[i] for i in state.getLegalActions() if
                                                     i != 'Stop'], self.explore_rate)

        reward = state.getScore() - self.prev_state.getScore()
        max_qval = self.get_qval_ref(q_state, self.agent_Q)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + self.decay * max_qval - prev_qval_ref[self.prev_action])
        self.prev_state = state
        self.prev_action = current_action
        if self.look_prob > 0:
            self.look_prob -= 1 / 100
        return self.prev_action


    def getUnConsistentAction(self, best_action, C=0.8):
        probs = [(1-C)/3]*4
        probs[best_action] = C
        return np.random.choice(4, p=probs)

    def get_action_q_update(self, state):
        if self.prev_state is None:
            legal = state.getLegalPacmanActions()
            if Directions.STOP in legal: legal.remove(Directions.STOP)
            successors = [(state.generateSuccessor(0, action), action) for action in legal]
            scored = [(self.evaluationFunction(state), action) for state, action in successors]
            bestScore = max(scored)[0]
            bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
            self.prev_state = state
            self.prev_action = self.direction2num[random.choice(bestActions)]
            return self.prev_action

        prev_qstate = self.get_qstate(self.prev_state)
        q_state = self.get_qstate(state)
        control_prob = random.random()
        teach = False
        if random.random() < self.L:
            teach = True

        reward = state.getScore() - self.prev_state.getScore()
        qval = self.get_qval_ref(q_state, self.agent_Q)
        if teach:
            for x in range(len(qval)):
                qval[x] += self.B * self.reward_bias(x, self.getTeacherGreedyAction(q_state), 0.8)

        current_action = self.getGreedyAction(q_state,
                                              [self.direction2num[i] for i in state.getLegalActions() if
                                               i != 'Stop'], self.explore_rate)

        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if
                                           i != 'Stop'])

        max_qval = qval[max_action]

        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + self.decay * max_qval - prev_qval_ref[self.prev_action])
        if teach:
            prev_qval_ref[self.prev_action] += self.B * self.reward_bias(self.prev_action, self.getTeacherGreedyAction(prev_qstate), 0.8)
        self.prev_state = state
        self.prev_action = current_action

        return self.prev_action

    def final(self, state):
        qval = self.get_qval_ref(self.get_qstate(self.prev_state), self.agent_Q)
        if state.isWin():
            reward = 500
        else:
            reward = -500
        qval[self.prev_action] += self.update_rate * (reward - qval[self.prev_action])
        self.explore_rate += 0.1 / self.total_episode
        if self.B > 0:
            self.B -= 1. / self.total_episode
        if self.B < 0:
            self.B = 0
        if self.explore_rate > 1:
            self.explore_rate = 1

        if self.count < self.total_episode:
            self.update_multi_bandit(state.getScore())
            self.need_change = True

        # print(self.count, state.getScore())
        self.count += 1
        self.score_list.append(1 if state.isWin() else 0)
        if self.count % 100 == 0:
            self.hundred_mean.append(np.sum(self.score_list))
            print(self.count / 100, self.hundred_mean[-1])
            self.score_list = []
        if self.count % 100 == 0:
            axes = plt.gca()
            axes.set_xlim([0, 120])
            axes.set_ylim([0, 100])
            plt.plot(list(range(0, len(self.hundred_mean))), self.hundred_mean)
            plt.show()
            self.sigint_handler()

    def reward_bias(self, select_action, best_action, C=1., rh=10):
        if select_action == best_action:
            if random.random() > C:
                return -rh
            else:
                return rh
        else:
            if random.random() > C:
                return rh
            else:
                return -rh

    def getGreedyAction(self, q_state, legal, explore_rate=1., action_bias=False):
        # just a copy of original self.Q
        q_value = self.get_qval_ref(q_state, self.agent_Q).copy()
        best_action = self.getTeacherGreedyAction(q_state)
        if action_bias:
            bias = self.action_bias(best_action)
            # beliefs = self.get_bval_ref(q_state)
            q_value += self.B * bias
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

    def action_bias(self, best_action, C=0.8, rh=10):
        probs = [(1 - C) / 3] * 4
        probs[best_action] = C
        unconsist_best_action = np.random.choice(4, p=probs)
        bias = [-rh] * 4
        bias[unconsist_best_action] = rh
        return np.array(bias)

    def get_qval_ref(self, q_state, Q):
        q_value = Q
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
