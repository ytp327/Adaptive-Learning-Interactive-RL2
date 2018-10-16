import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib as mpl

mpl.style.use('seaborn')

normal_all_episodes = pickle.load(open("nohumanintroduction.dat", "rb"))
multi_all_episodes = pickle.load(open("multi_humaninstruction.dat", "rb"))
mb_all_episodes = pickle.load(open("MBL0.01C0.8.dat","rb"))
ab_all_episodes = pickle.load(open("humanintroduction_actionbiasing.dat", "rb"))
abwithout_all_episodes = pickle.load(open("abL0.01C0.8.dat", "rb"))


normal_avg = []
multi_avg = []
mb_avg = []
ab_avg = []
abwithout_avg = []

for i in range(30000):
    normal_avg.append(0)
    for episode in normal_all_episodes:
        normal_avg[i] += episode[i]
    normal_avg[i] /= len(normal_all_episodes)

avg = []
for i in range(len(normal_avg)):
    avg.append(np.average(normal_avg[i:i+1000]))

avg = [avg[i] for i in range(len(avg)) if not i % 1000 ]
plt.plot([i for i in range(len(normal_avg)-1000) if not i % 1000], avg[:-1], 'o-')
for i in range(30000):
    multi_avg.append(0)
    for episode in multi_all_episodes:
        multi_avg[i] += episode[i]
    multi_avg[i] /= len(multi_all_episodes)
avg = []
for i in range(len(multi_avg)):
    avg.append(np.average(multi_avg[i:i+1000]))

avg = [avg[i] for i in range(len(avg)) if not i % 1000 ]
plt.plot([i for i in range(len(multi_avg)-1000) if not i % 1000], avg[:-1], 'o-')

for i in range(30000):
    mb_avg.append(0)
    for episode in mb_all_episodes:
        mb_avg[i] += episode[i]
    mb_avg[i] /= len(mb_all_episodes)

print(len(mb_all_episodes))
avg = []
for i in range(len(mb_avg)):
    avg.append(np.average(mb_avg[i:i+1000]))

avg = [avg[i] for i in range(len(avg)) if not i % 1000 ]
plt.plot([i for i in range(len(normal_avg)-1000) if not i % 1000], avg[:-1], 'o-')


for i in range(30000):
    ab_avg.append(0)
    for episode in ab_all_episodes:
        ab_avg[i] += episode[i]
    ab_avg[i] /= len(ab_all_episodes)

avg = []
for i in range(len(ab_avg)):
    avg.append(np.average(ab_avg[i:i+1000]))

avg = [avg[i] for i in range(len(avg)) if not i % 1000 ]
plt.plot([i for i in range(len(normal_avg)-1000) if not i % 1000], avg[:-1], 'o-')

for i in range(30000):
    abwithout_avg.append(0)
    for episode in abwithout_all_episodes:
        abwithout_avg[i] += episode[i]
        abwithout_avg[i] /= len(abwithout_all_episodes)

avg = []
for i in range(len(abwithout_avg)):
    avg.append(np.average(abwithout_avg[i:i+1000]))

avg = [avg[i] for i in range(len(avg)) if not i % 1000 ]
# plt.plot([i for i in range(len(normal_avg)-1000) if not i % 1000], avg[:-1], 'o-')


plt.legend(['Q Learning_multi_degenerate','multi_learning','multi_without_instruction','action_biasing_with_instruction'], loc = 0, ncol = 1)
plt.xlabel('Numbers of episodes')
plt.ylabel('Average Reward')
plt.show()
