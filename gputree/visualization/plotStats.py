import matplotlib.pyplot as plt

fp = open('../data/depthData.txt', 'r')

newline = fp.readline()
numPoints = []
numDim = []
runTime = []

while newline:
    token = newline.split(' ')
    numPoints.append(int(token[0]))
    numDim.append(float(token[1]))
    runTime.append(int(token[1]))
    newline = fp.readline()

print(numPoints)
plt.subplot(2, 2, 1)
plt.xscale('log')
plt.title('Tree Depth over Different Data Size')
plt.plot(numPoints[0:5], runTime[0:5])
plt.xlabel('Number of Points')
plt.ylabel('Tree Depth')

plt.show()

fp.close()
