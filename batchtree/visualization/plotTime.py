import matplotlib.pyplot as plt

fp = open('../graphs/cpu_gpu_total.txt')

queries = []

line = fp.readline().split()
cputime = []

while(line):
    if line[0][0] != '#':
        queries.append(int(line[0]))
        cputime.append(float(line[1]))
    line = fp.readline().split()

line = fp.readline().split()
gputime = []

while(line):
    if line[0][0] != '#':
        gputime.append(float(line[1]))
    line = fp.readline().split()

# line = fp.readline().split()
# batch_gputime = []

# while(line):
#     if line[0][0] != '#':
#         batch_gputime.append(float(line[1]))
#     line = fp.readline().split()

# plt.plot(queries, cputime)
plt.xscale('log', basex=2)
plt.plot(queries, cputime)
plt.plot(queries, gputime)
# plt.plot(queries, batch_gputime)
# plt.axvline(x=512, color='red')
plt.legend(['CPU', 'GPU'])
plt.title('Total time taken vs Number of query points')
plt.xlabel('Query points')
plt.ylabel('Time taken (s)')
plt.grid()
plt.show()