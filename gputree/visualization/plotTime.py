import matplotlib.pyplot as plt

cpu_f = open('../data/cpu_time.txt')
gpu_f = open('../data/data_timeTaken.txt')

queries = []

line = cpu_f.readline()
cputime = []

while(line):
    cputime.append(float(line))
    line = cpu_f.readline()

line = gpu_f.readline()
gputime = []

while(line):
    gputime.append(float(line))
    line = gpu_f.readline()

for i in range(len(gputime)):
    queries.append(2 ** (3 + i))

plt.plot(queries, cputime)
plt.plot(queries, gputime)
plt.show()