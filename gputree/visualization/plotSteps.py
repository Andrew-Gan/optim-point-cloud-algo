import matplotlib.pyplot as plt

fp = open('../data/data_steps.txt')

newline = fp.readline()
steps = []

while newline:
    token = int(newline)
    steps.append(token)
    newline = fp.readline()
factors = []
j = 0.1
while j < 1.01:
    factors.append(j)
    j += 0.01

avgSteps = []

for i in range(0, len(steps), 10):
    avgSteps.append(sum(steps[i:i+10]) / 10)
plt.title('Cost to find Nearest Neighbor at Enlargement = 0.2')
plt.xlabel('Rou Factor')
plt.ylabel('Points calculated')
plt.plot(factors, avgSteps)
plt.show()