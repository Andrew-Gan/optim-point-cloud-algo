import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# expected_fp = open('../expected.txt', 'r')
# newline_e = expected_fp.readline()

# newline_str = []

# while(newline_e):
#     newline_str.append(newline_e)
#     newline_e = expected_fp.readline()

# for i in range(85):
#     if(i != 12):
#         digStr = "{0:0=3d}".format(i)
#         actual_fp = open('../data/data_nearestPoint' + digStr)
#         newline_a = actual_fp.readline()

#         trueCount = 0
#         counter = 0

#         while(newline_a):
#             trueCount += (newline_str[counter] == newline_a)
#             newline_a = actual_fp.readline()
#             counter += 1
        
#         actual_fp.close()

#         print('got {} out of {} queries correct at {:.4f}%'.format(trueCount, len(newline_str), 100 * (trueCount / len(newline_str))))

fp = open('../visualization/sp_accu.txt', 'r')

newline = fp.readline()

rho = []
accu = []

i = 0
while i < 0.84:
    rho.append(i)
    i += 0.01

del rho[12]

while newline:
    accu.append(float(newline))
    newline = fp.readline()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(9, min_n_ticks=9, integer=True))

# plt.plot(queries, cputime)
plt.plot(rho, accu)
plt.title('Accuracy of Nearest Points vs Rho with data size = 10000, query size = 32768')
plt.xlabel('Rho value ρ')
plt.ylabel('Accuracy')
plt.grid()
plt.show()