import numpy as np
import joblib
from sklearn.datasets import fetch_openml
import sys

"""----------------------------------------------------------------------------------------------------------"""
# Fetching the dataset from server and scale
# it for our purpose
#

# Fetch the datasets

mnist = fetch_openml("mnist_784")

# Normalize the dataset
mnist_data = mnist.data.values / np.max(mnist.data.values)

# Get Targets
mnist_target = mnist.target.astype(int)

print("Data Ready", file = sys.stderr , flush=True)

# Convert labels to binaries
mnist_label = np.zeros((mnist_data.shape[0], 10))
for idx in range(mnist_data.shape[0]):
    mnist_label[idx,int(mnist.target[idx])] = 1

# Clean memory space of mnist
mnist = None

def y_of_x(x: np.ndarray, W_E_xy: np.ndarray = None,
           W_I_yy: np.ndarray = None, noise=None
           ):
    if W_E_xy is None:
        return 0.5 * (x + np.abs(x))

    if len(x.shape) == 1:
        y = np.zeros((W_E_xy.shape[0]))
    else:
        y = np.zeros((W_E_xy.shape[0], x.shape[1]))

    for t in range(20):
        input4y = np.matmul(W_E_xy, x) / np.sqrt(W_E_xy.shape[1])
        if W_I_yy is not None:
            input4y -= np.matmul(W_I_yy, (y)) / np.sqrt(W_I_yy.shape[1])
        if noise is not None:
            input4y += noise
        y += (-y + np.maximum(input4y, 0)) / np.sqrt(5.0)

    return y

def update_w(x, w, limit=False, ltd=True, mature=True, compet=True, rate=0.1):
    #limit=True, ltd=False: ltp+sc   |   limit=False, ltd=True: ltp+ltd   |   limit=False, ltd=False: ltp
    #rate = learning rate

    w_local = np.copy(w)

    for idx in range(x.shape[0]):

        xt = x[idx]

        noise = np.random.normal(size=w_local.shape[0])

        y = y_of_x(xt, w_local, noise=noise)

        if ltd:

            delta_w = np.outer(y, xt - 0.25)

            w_local += rate * delta_w

            for i in range(w_local.shape[0]):
                if np.max(w_local[i]) >= 1.0:
                    w_local[i] /= np.max(w_local[i])

        elif limit:

            delta_w = np.outer(y, xt)
            if mature:

                for i in range(w_local.shape[0]):
                    if np.sum(w_local[i]) >= 0.1:
                        delta_w[i] = np.zeros_like(delta_w[i])

            if compet:

                for i in range(delta_w.shape[1]):
                    delta_w[:, i] -= np.mean(delta_w[:, i])

            w_local += rate * delta_w

            for i in range(w_local.shape[1]):
                if np.max(w_local[:, i]) > 1:
                    w_local[:, i] /= np.max(w_local[:, i])

            for i in range(w_local.shape[0]):
                if np.max(w_local[i]) < 0:
                    w_local[i] = np.zeros_like(w_local[i])

        else:

            delta_w = np.outer(y, xt)

            w_local += rate * delta_w

            for i in range(w_local.shape[0]):
                if np.max(w_local[i]) > 1:
                    w_local[i] /= np.max(w_local[i])

    return w_local

def cal_fit_rate(W_E_xy: np.ndarray,W_E_yy:np.ndarray,
                 data_sample: np.ndarray, target_sample: np.ndarray,
                 data_test: np.ndarray, target_test: np.ndarray):

    y = y_of_x(data_sample.transpose(), W_E_xy)#(784,n_training)(n_l1,784)=(n_l1,n_training)

    y_l = y_of_x(y,W_E_yy)#(n_l1,n_training)(n_l2,n_l1)=(n_l2,n_training)

    y_l_tmp = np.append(y_l, np.ones((1, y_l.shape[1])), axis=0)#(n_l2 +1,n_training)

    y_l = y_l_tmp

    W_yz = np.matmul(y_l, y_l.transpose())#(n_l2 +1,n_training)(n_training,n_l2 +1)=(n_l2 +1,n_l2 +1)

    W_yz = np.matmul(y_l.transpose(), np.linalg.pinv(W_yz))#(n_training,n_l2 +1)(n_l2 +1,n_l2 +1)=(n_training,n_l2 +1)

    W_yz = np.matmul(target_sample.transpose(), W_yz)#(10，n_training)(n_training,n_l2 +1)=(10,n_l2 +1)

    y = y_of_x(data_test.transpose(), W_E_xy)#(784,n_test)(n_l1,784)=(n_l1,n_test)

    y_l = y_of_x(y,W_E_yy)#(n_l1,n_test)(n_l2,n_l1)=(n_l2,n_test)

    y_l_tmp = np.append(y_l, np.ones((1, y_l.shape[1])), axis=0)#(n_l2 +1,n_test)

    y_l = y_l_tmp#(n_l2 +1,n_test)

    z = np.matmul(W_yz, y_l)#(10,n_l2 +1)(n_l2 +1,n_test)=(10,n_test)

    ans = 1.0 - np.sum(np.heaviside(
        np.absolute(z.argmax(axis=0) - target_test.argmax(axis=1)) - 0.5
        , 0)) / float(data_test.shape[0])

    return ans

def local_loop(mnist_data, mnist_label, N_train_sample, N_test, N_y1, N_y2):
    random_list = np.arange(mnist_data.shape[0])

    np.random.shuffle(random_list)

    mnist_target_training = mnist_label[random_list[:N_train_sample]]
    mnist_data_training = mnist_data[random_list[:N_train_sample]]

    mnist_target_testing = mnist_label[random_list[(mnist_data.shape[0] - N_test):]]
    mnist_data_testing = mnist_data[random_list[(mnist_data.shape[0] - N_test):]]

    W_E_xy = np.zeros((N_y1, mnist_data.shape[1]))#(n_l1,784)

    W_E_xy = update_w(mnist_data_training, W_E_xy,limit=True, ltd=False)#(n_training,784)(n_l1,784) sc

    data_layer = y_of_x(mnist_data_training.transpose(),W_E_xy)#(n_l1,n_training)

    W_E_yy = np.zeros((N_y2,data_layer.shape[0]))#(n_l2,n_l1)

    W_E_yy = update_w(data_layer.transpose(),W_E_yy,limit=False, ltd=False)#(n_training,n_l1)(n_l2,n_l1) ltp

    return cal_fit_rate(W_E_xy, W_E_yy, mnist_data_training, mnist_target_training,
                        mnist_data_testing, mnist_target_testing)

for N_y1 in [100, 500, 1000]:
    for N_y2 in [100, 500, 1000]:
        for N_train_sample in [100, 500, 1000, 10000]:
            N_test =int(mnist_data.shape[0] * 0.1)
            sample_ans = []
            sample_ans = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(local_loop)(mnist_data, mnist_label, N_train_sample, N_test, N_y1,N_y2)
                for idx in range(10))

            print(N_y1,N_y2, N_train_sample, np.mean(sample_ans), np.std(sample_ans), flush=True)

    print(flush=True)
