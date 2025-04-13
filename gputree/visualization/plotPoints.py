# only for 2D points, duh

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fp = open('../data/data_pointList.txt', 'r')
qfp = open('../data/data_nearestPointList.txt', 'r')
# sfp = open('../data/data_subspace.txt', 'r')

fp.readline()
newline = fp.readline()
if(len(newline.split(' ')) != 3):
    print('Error: Not 2 dimensions')
    exit()
x = []
y = []
while not 'Query' in newline:
    token = newline.split(' ')
    x.append(int(token[0]))
    y.append(int(token[1]))
    newline = fp.readline()

qx = []
qy = []
newline = fp.readline()
while newline:
    token = newline.split(' ')
    qx.append(int(token[0]))
    qy.append(int(token[1]))
    newline = fp.readline()

fp.close()

nnx = []
nny = []
newline = qfp.readline()
while newline:
    token = newline.split(' ')
    nnx.append(int(token[0]))
    nny.append(int(token[1]))
    newline = qfp.readline()

qfp.close()

# subx = []
# suby = []
# newline = sfp.readline()
# # if(len(nnx) == 1):
# while newline:
#     token = newline.split(' ')
#     subx.append(float(token[0]))
#     suby.append(float(token[1]))
#     newline = sfp.readline()

# sfp.close()

# pre-plot check
if(len(x) != len(y) or len(qx) != len(nnx)):
    print('Error unequal array length for plotting')
    exit()

fig, ax = plt.subplots(1)
plt.scatter(x, y)           # plot data points
plt.scatter(qx, qy)         # plot query points
plt.scatter(nnx, nny)       # plot nearest neighbors
plt.legend(['Data points', 'Query points', 'Nearest Neighbours'])
for i in range(len(qx)):
    plt.plot([qx[i], nnx[i]], [qy[i], nny[i]], color='orange')
# for j in range(0, len(subx), 2):
#     ax.add_patch(Rectangle((subx[j], suby[j]), subx[j+1]-subx[j], suby[j+1]-suby[j], fill=False, color='red'))
plt.show()