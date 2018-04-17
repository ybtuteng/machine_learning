import math

ALPHA = 0.3
DIFF = 0.000001


def predict(theta, data):
    results = []
    for i in range(0, data.__len__()):
        temp = 0
        for j in range(1, theta.__len__()):
            temp += theta[j] * data[i][j - 1]
        temp = 1 / (1 + math.e ** (-1 * (temp + theta[0])))
        results.append(temp)
    return results


def training(training_data):
    size = training_data.__len__()
    dimension = training_data[0].__len__()
    hxs = []
    theta = []
    for i in range(0, dimension):
        theta.append(1)
    initial = 0
    for i in range(0, size):
        hx = theta[0]
        for j in range(1, dimension):
            hx += theta[j] * training_data[i][j]
        hx = 1 / (1 + math.e ** (-1 * hx))
        hxs.append(hx)
        initial += (-1 * (training_data[i][0] * math.log(hx) + (1 - training_data[i][0]) * math.log(1 - hx)))
    initial /= size
    iteration = initial
    initial = 0
    counts = 1
    while abs(iteration - initial) > DIFF:
        print("第", counts, "次迭代, diff=", abs(iteration - initial))
        initial = iteration
        gap = 0
        for j in range(0, size):
            gap += (hxs[j] - training_data[j][0])
        theta[0] = theta[0] - ALPHA * gap / size
        for i in range(1, dimension):
            gap = 0
            for j in range(0, size):
                gap += (hxs[j] - training_data[j][0]) * training_data[j][i]
            theta[i] = theta[i] - ALPHA * gap / size
            for m in range(0, size):
                hx = theta[0]
                for j in range(1, dimension):
                    hx += theta[j] * training_data[i][j]
                hx = 1 / (1 + math.e ** (-1 * hx))
                hxs[i] = hx
                iteration += -1 * (training_data[i][0] * math.log(hx) + (1 - training_data[i][0]) * math.log(1 - hx))
            iteration /= size
        counts += 1
    print('training done,theta=', theta)
    return theta


if __name__ == '__main__':
    training_data = [[1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1],
                     [0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1]]
    test_data = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]
    theta = training(training_data)
    res = predict(theta, test_data)
    print(res)
