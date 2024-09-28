import numpy as np
import time
import xlearn as xl
import matplotlib.pyplot as plt
import os


def load_fm_model(model_txt, N, Nf):
    model = model_txt
    idx = 1
    idf = 1
    offset = []
    L = []
    V = [[] for _ in range(N)]
    with open(model, 'r') as f:
        for line in f:
            split_line = line.strip().split(' ')
            numbers = list(map(float, split_line[1:]))
            if idx == 1:
                offset.append(numbers)
            elif 1 < idx <= 1 + N:
                L.append(numbers)
            elif idx > 1 + N:
                V[idf - 1].append(numbers)
                if len(V[idf - 1]) == Nf:
                    idf += 1
            idx += 1

    offset = np.array(offset)
    L = np.array(L)
    V = np.array(V)

    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Q[i, j] = L[i]
            elif i < j:
                Q[i, j] = np.dot(V[i, j % Nf], V[j, i % Nf])
            else:
                Q[i, j] = 0
    return Q, offset


def train(model, param, trainSet, validSet, model_txt, model_out, restart=False):
    t_start = time.time()
    model.setTrain(trainSet)
    model.setValidate(validSet)
    if restart and os.path.exists(model_out):
        model.setPreModel(model_out)
    model.setTXTModel(model_txt)
    model.disableNorm()
    model.fit(param, model_out)
    t_end = time.time()
    print("total ffm train wall time: " + str(t_end - t_start))


def infer(model, testSet, model_out, testResult, fig=False):
    model.setTest(testSet)  # set test set for fm model.
    model.predict(model_out, testResult)
    e_pred = []
    with open(testResult, 'r') as f:
        for line in f:
            e_pred.append(float(line))
    e = []
    with open(testSet, 'r') as f:
        for line in f:
            e.append(float(line.strip().split(' ')[0]))
    if fig:
        plt.scatter(e, e_pred)
        plt.savefig("evaluation.png")
    return e_pred, e


