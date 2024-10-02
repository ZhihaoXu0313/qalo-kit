import numpy as np
import time
import xlearn as xl
import matplotlib.pyplot as plt
import os
import random


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


def fm_train(model, param, trainSet, validSet, model_txt, model_out, restart=False):
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


def fm_infer(model, testSet, model_out, testResult, fig=False):
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


def split_libffm_data(fmpath, sample_ratio):
    with open(os.path.join(fmpath, "libffm_data_total.txt"), 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    n_pick = int(len(lines) * sample_ratio)
    select_data = lines[:n_pick]
    with open(os.path.join(fmpath, "train_ffm.txt"), "w") as file:
        file.writelines(select_data[:int(0.8 * n_pick)])
    with open(os.path.join(fmpath, "valid_ffm.txt"), "w") as file:
        file.writelines(select_data[int(0.8 * n_pick):int(0.9 * n_pick)])
    with open(os.path.join(fmpath, "test_ffm.txt"), "w") as file:
        file.writelines(select_data[int(0.9 * n_pick):])
            