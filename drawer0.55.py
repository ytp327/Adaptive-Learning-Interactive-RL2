import matplotlib.pyplot as plt
import numpy as np
import pickle

normal_all_episodes = pickle.load(open("normalq.dat", "rb"))
ab_all_episodes = pickle.load(open("abq.dat", "rb"))
rs_all_episodes = pickle.load(open("rsq.dat", "rb"))
cs_all_episodes = pickle.load(open("csq.dat", "rb"))
qu_all_episodes = pickle.load(open("quq.dat", "rb"))
mb_all_episodes = pickle.load(open("MBq.dat", "rb"))
rs_avg = []
normal_avg = []
ab_avg = []
cs_avg = []
qu_avg = []
mb_avg = []

for i in range(len(normal_all_episodes[0])):
    normal_avg.append(0)
    for episode in normal_all_episodes:
        normal_avg[i] += episode[i]
    normal_avg[i] /= len(normal_all_episodes)

avg = []
for i in range(len(normal_avg)):
    avg.append(np.average(normal_avg[i:i+1000]))

plt.plot(list(range(len(avg)-15000)), avg[:-15000])

for i in range(len(ab_all_episodes[0])):
    ab_avg.append(0)
    for episode in ab_all_episodes:
        ab_avg[i] += episode[i]
    ab_avg[i] /= len(ab_all_episodes)

avg = []
for i in range(len(ab_avg)):
    avg.append(np.average(ab_avg[i:i+1000]))

plt.plot(list(range(len(avg)-15000)), avg[:-15000])

for i in range(len(rs_all_episodes[0])):
    rs_avg.append(0)
    for episode in rs_all_episodes:
        rs_avg[i] += episode[i]
    rs_avg[i] /= len(rs_all_episodes)

avg = []
for i in range(len(rs_avg)):
    avg.append(np.average(rs_avg[i:i+1000]))

plt.plot(list(range(len(avg)-15000)), avg[:-15000])

for i in range(len(cs_all_episodes[0])):
    cs_avg.append(0)
    for episode in cs_all_episodes:
        cs_avg[i] += episode[i]
    cs_avg[i] /= len(cs_all_episodes)

avg = []
for i in range(len(cs_avg)):
    avg.append(np.average(cs_avg[i:i+1000]))

plt.plot(list(range(len(avg)-15000)), avg[:-15000])

for i in range(len(qu_all_episodes[0])):
    qu_avg.append(0)
    for episode in qu_all_episodes:
        qu_avg[i] += episode[i]
    qu_avg[i] /= len(qu_all_episodes)

avg = []
for i in range(len(qu_avg)):
    avg.append(np.average(qu_avg[i:i+1000]))

plt.plot(list(range(len(avg)-15000)), avg[:-15000])

for i in range(15000):
    mb_avg.append(0)
    for episode in mb_all_episodes:
        mb_avg[i] += episode[i]
    mb_avg[i] /= len(mb_all_episodes)

print(len(mb_all_episodes))
avg = []
for i in range(len(mb_avg)):
    avg.append(np.average(mb_avg[i:i+1000]))

plt.plot(list(range(len(avg))), avg)
plt.legend(['Q Learning', 'AB', 'RS', 'CS', 'QU', 'MB'], loc = 0, ncol = 3)
plt.show()
