import numpy as np


def loss(y_true, y_false):
    int_true = np.square(y_true[:,0])
    int_true += np.square(y_true[:,1])
    int_true += np.square(y_true[:,2])
    int_pred = np.square(y_pred[:,0])
    int_pred += np.square(y_pred[:,1])
    int_pred += np.square(y_pred[:,2])
    res = np.square(int_true - int_pred)
    res = np.sqrt(res)
    res = np.mean(res)
    return res


if __name__ in "__main__":
    y_true = []
    y_true.append([0,0,1])
    y_true.append([-1, -1, 2])

    y_pred = []
    y_pred.append([0, 0, 1.5])
    y_true.append([-1, -1, 2])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(loss(y_true,
               y_pred))


