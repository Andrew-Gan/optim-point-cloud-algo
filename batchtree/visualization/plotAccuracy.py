import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# expected_fp = open('../expected.txt', 'r')
# newline_e = expected_fp.readline()

# newline_str = []

# while(newline_e):
#     newline_str.append(newline_e)
#     newline_e = expected_fp.readline()

# for i in range(20):
#     digStr = "{0:0=2d}".format(i)
#     actual_fp = open('../data/data_nearestPointList' + digStr)
#     newline_a = actual_fp.readline()

#     trueCount = 0
#     counter = 0

#     while(newline_a):
#         trueCount += (newline_str[counter] == newline_a)
#         newline_a = actual_fp.readline()
#         counter += 1
    
#     actual_fp.close()

#     print('got {} out of {} queries correct at {:.4f}%'.format(trueCount, len(newline_str), 100 * (trueCount / len(newline_str))))

fp = open('../graphs/batch_accu_time.txt', 'r')

newline = fp.readline()

truncateNode = []
time = []
accu = []

while newline:
    row = newline.split()
    truncateNode.append(int(row[0]))
    time.append(float(row[1]))
    accu.append(float(row[2]))
    newline = fp.readline()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(15, min_n_ticks=15, integer=True))

# plt.plot(queries, cputime)
plt.plot(truncateNode, time)
plt.title('Batch Processing: Time Taken vs Number of Truncated Nodes for data size = 10000, query size = 32768')
plt.xlabel('Truncated Nodes')
plt.ylabel('Time taken (s)')
plt.grid()
plt.show()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(15, min_n_ticks=15, integer=True))

plt.plot(truncateNode, accu)
plt.title('Batch Processing: Accuracy vs Number of Truncated Nodes for data size = 10000, query size = 32768')
plt.xlabel('Truncated Nodes')
plt.ylabel('Accuracy')
plt.grid()
plt.show()