import sys
import matplotlib.pyplot as plt

def kNN(points, query):
    currentShortest = points[0]
    currentMin = sys.float_info.max
    for i in range(len(points)):
        dist = 0
        for d in range(len(points[i])):
            dist += (points[i][d] - query[d])**2
        if(dist < currentMin):
            currentShortest = points[i]
            currentMin = dist
    return currentShortest

def readProducedOutput(filepath):
    fp_output = open(filepath, 'r')
    newline = fp_output.readline()
    algo_out = []
    rou_arr = []
    while(newline):
        token = newline.rstrip().split(' ')
        if(len(token) == 1):
            rou_arr.append(float(token[0]))
        else:
            tmp = []
            for d in token:
                tmp.append(int(d))
            algo_out.append(tuple(tmp))
            tmp.clear()
        newline = fp_output.readline()
    return rou_arr, algo_out
        

def verifyAccuracy(correct_out, algo_out):
    if(len(correct_out) != len(algo_out)):
        print('Error: unequal size detected in verifyAccuracy()')
        return
    correctCount = 0
    for i in range(len(correct_out)):
        if(correct_out[i] == algo_out[i]):
            correctCount += 1
    return correctCount / (i + 1)

def plotAccuracy(rou_arr, time_arr, acc_arr):
    plt.subplot(2, 1, 1)
    plt.title('kNN Accuracy vs Rou Factor when Enlargement Factor = 0.2')
    plt.xlabel('Rou Factor')
    plt.ylabel('Accuracy (%)')
    plt.plot(rou_arr, acc_arr)
    plt.subplot(2, 1, 2)
    plt.title('kNN Efficiency vs Rou Factor when Enlargement Factor = 0.2')
    plt.xlabel('Rou Factor')
    plt.ylabel('Time taken (s)')
    plt.plot(rou_arr, time_arr)
    plt.show()

if __name__ == '__main__':
    fp_points = open('../data/data_pointList.txt', 'r')
    fp_points.readline()
    newline = fp_points.readline()
    points = []
    queries = []
    correct_out = []
    acc_arr = []

    while('Query' not in newline):
        token = newline.rstrip().split(' ')
        tmp = []
        for d in range(len(token)):
            tmp.append(int(token[d]))
        points.append(tuple(tmp))
        newline = fp_points.readline()

    newline = fp_points.readline()
    while(newline):
        token = newline.rstrip().split(' ')
        tmp = []
        for d in range(len(token)):
            tmp.append(int(token[d]))
        queries.append(tuple(tmp))
        newline = fp_points.readline()
    for i in range(len(queries)):
        correct_out.append(kNN(points, queries[i]))
    rou_arr, algo_out = readProducedOutput('../data/data_nearestPointList.txt')
    
    #for i in range(len(rou_arr)):
    for i in range(len(rou_arr)):
        acc_arr.append(verifyAccuracy(correct_out, algo_out[i*100:i*100+100]))
    
    time_arr = []
    fp_time = open('../data/data_timeTaken.txt', 'r')
    newline = fp_time.readline()
    while(newline):
        time_arr.append(float(newline.rstrip().split(' ')[1]))
        newline = fp_time.readline()
    plotAccuracy(rou_arr, time_arr, acc_arr)