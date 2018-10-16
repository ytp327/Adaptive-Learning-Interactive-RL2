from pacman import Directions
from game import Agent
import random
import util
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import msvcrt
TEACHER_Q_PATH = 'Q_finish.pkl'
STUDENT_Q_PATH = 'agentQCS.pkl'

class CSAgent(Agent):

    def __init__(self, evalFn="scoreEvaluation", **args):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None
        self.L = 0.01
        self.C = 0.8




        if os.path.exists('ControlSharingTrueHuman.dat'):
            with open('ControlSharingTrueHuman.dat', 'rb') as f:
                self.all_episode = pickle.load(f)
        else:
            self.all_episode = []
        self.beQuiet = 0
        self.stop = 0
        self.veryslow = 0





        self.direction2num = {"West": 0, "East": 1, "North": 2, "South": 3, "Stop": 4}
        self.num2direction = ["West", "East", "North", "South", "Stop"]
        self.init()
        self.episode_max = 30000
        # print('run ', self.count / self.episode_max + 1)




    def kbhitquiet(self):
        if not self.beQuiet:
            raise Exception('kbhitquiet should be called when not quiet')
        self.totalll += 1.0
        self.thereis_guidance = 0
        while msvcrt.kbhit():
            a = ord(msvcrt.getch())
            if a == 115:
                if self.veryslow:
                    self.veryslow = 0
                else:
                    self.veryslow = 1
                self.beQuiet = 0  # SLOW
            if a == 102:
                self.beQuiet = 1  # FAST
                self.veryslow = 1
            if a == 100:
                self.stop = 1  # d stop
            if a == 0 or a == 224:
                b = ord(msvcrt.getch())
        if self.stop == 1:
            a = 0
            while a != 101:
                if msvcrt.kbhit():
                    a = ord(msvcrt.getch())
            self.stop = 0






    def init(self):






        self.ll = 0
        self.totalll = 0
        self.stop = 0
        self.veryslow = 0
        self.bestaction = 0
        self.scoretimes = 0
        self.thereis_guidance = 0
        self.flag = -1






        self.agent_Q = self.create_qtable("empty")
        self.prev_state = None
        self.prev_action = None
        self.update_rate = 0.3
        self.episode_score = []
        self.explore_rate = 0.90
        self.B = 1
        self.count = 0
        # self.score_list = []
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

    # def sigint_handler(self):
    #     with open(STUDENT_Q_PATH, 'wb') as f:
    #         pickle.dump(self.agent_Q, f)
    #     import sys
    #     if self.count > 300000:
    #         sys.exit(0)




    def returnbeQuiet(self):
        return self.beQuiet




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
        prev_qstate = self.get_qstate(self.prev_state)
        q_state = self.get_qstate(state)
        max_action = self.getGreedyAction(q_state,
                                          [self.direction2num[i] for i in state.getLegalActions() if
                                           i != 'Stop'])

        if not self.beQuiet:
            best_action = self.getTeacherGreedyAction(q_state)
            current_action = best_action
            legal = [self.direction2num[i] for i in state.getLegalActions() if i != 'Stop']
            if current_action not in legal:
                current_action = self.direction2num[[i for i in state.getLegalActions() if i != 'Stop'][0]]
            if random.random() > self.B:
                current_action = self.getGreedyAction(q_state,
                                                      [self.direction2num[i] for i in state.getLegalActions() if
                                                       i != 'Stop'], self.explore_rate)
        else:
            self.kbhitquiet()
            current_action = self.getGreedyAction(q_state,
                                                    [self.direction2num[i] for i in state.getLegalActions() if
                                                     i != 'Stop'], self.explore_rate)

        reward = state.getScore() - self.prev_state.getScore()
        max_qval = self.get_qval_ref(q_state, self.agent_Q)[max_action]
        prev_qval_ref = self.get_qval_ref(prev_qstate, self.agent_Q)
        prev_qval_ref[self.prev_action] += self.update_rate * (reward + 0.7 * max_qval - prev_qval_ref[self.prev_action])
        self.prev_state = state
        self.prev_action = current_action
        return self.num2direction[self.prev_action]





    def save(self):
        with open('ControlSharingTrueHuman.dat' , 'wb') as f:
            pickle.dump(self.all_episode, f)




    def final(self, state):
        prev_qstate = self.get_qstate(self.prev_state)
        qval = self.get_qval_ref(prev_qstate, self.agent_Q)
        if state.isWin():
            reward = 500
        else:
            reward = -500
        qval[self.prev_action] += self.update_rate * (reward - qval[self.prev_action])

        if self.explore_rate < 1:
            self.explore_rate += 0.1 / self.episode_max # 10000

        if self.explore_rate > 1:
            self.explore_rate = 1

        if self.B > 0:
            self.B -= 1 / self.episode_max
        if self.B < 0:
            self.B = 0




        self.scoretimes += 1 if state.isWin() else 0






        # print(self.count, state.getScore())
        self.count += 1
        # self.score_list.append(1 if state.isWin() else 0)













        if self.count % 2000 == 0 and self.totalll != 0:
            print self.ll/self.totalll
            print self.count / 100
        if self.count % 100 == 0:
            if self.scoretimes >10 and self.flag < 1:
                self.flag = 1
                print 'more than 10'
            if self.scoretimes >20 and self.flag <2:
                self.flag = 2
                print 'more than 20'
            if self.scoretimes >30 and self.flag <3:
                self.flag = 3
                print 'more than 30'
            if self.scoretimes >40 and self.flag <4:
                self.flag = 4
                print 'more than 40'
            if self.scoretimes >50 and self.flag <5:
                self.flag = 5
                print 'more than 50'
            if self.scoretimes > 60 and self.flag <6:
                self.flag = 6
                print 'more than 60'
            if self.scoretimes >70 and self.flag <7:
                self.flag = 7
                print 'more than 70'
            if self.scoretimes >80and self.flag <8:
                self.flag = 8
                print 'more than 80'
            if self.scoretimes >90 and self.flag <9:
                self.flag = 9
                print 'more than 90'
            if self.scoretimes >= 99 and self.flag <10:
                self.flag = 10
                print 'succeed in 99'

            self.hundred_mean.append(self.scoretimes)
            self.scoretimes = 0
            # print(self.count / 100, self.hundred_mean[-1])











        self.episode_score.append(state.getScore())
        if self.count % self.episode_max == 0 and self.count != 0:
            self.all_episode.append(self.episode_score)
            self.init()
            self.save()
            print('run ', len(self.all_episode))


        # if self.count % 5000 == 0:
        #     plt.plot(list(range(0, len(self.hundred_mean))), self.hundred_mean)
        #     plt.show()
        #     self.sigint_handler()

    def getGreedyAction(self, q_state, legal, explore_rate=1):
        # just a copy of original self.Q
        q_value = self.get_qval_ref(q_state, self.agent_Q).copy()
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








    def getTeacherGreedyAction(self,q_state):
        self.thereis_guidance = 0
        if self.beQuiet == 0:
            best_action = self.bestaction
            self.thereis_guidance = 1
            self.ll += 1.0
        self.totalll += 1.0
        if self.beQuiet:
            raise Exception('when quiet , do not need teacher')
        while msvcrt.kbhit():
            a = ord(msvcrt.getch())
            if a == 115:
                if self.veryslow:
                    self.veryslow = 0
                else:
                    self.veryslow = 1
                self.beQuiet = 0  # SLOW
            if a == 102:
                self.beQuiet = 1  # FAST
                self.veryslow = 1
            if a == 100:
                self.stop = 1  # d stop
            if a == 0 or a == 224:
                b = ord(msvcrt.getch())
                if self.beQuiet:
                    continue
                x = a + (b * 256)  # -> 19936  right east 1  <-19424   left west 0
                # print(x)         # up   north  18656    2        down   south  20704 3
                if x == 19936 :
                    best_action = 1
                    self.bestaction = best_action
                elif x == 19424 :
                    best_action = 0
                    self.bestaction = best_action
                elif x == 18656 :
                    best_action = 2
                    self.bestaction = best_action
                elif x == 20704 :
                    best_action = 3
                    self.bestaction = best_action
        if self.stop == 1:
            a = 0
            while a != 101:
                if msvcrt.kbhit():
                    a = ord(msvcrt.getch())
            self.stop = 0
        a = self.ll/self.totalll
        #printllopen print a
        return best_action










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

