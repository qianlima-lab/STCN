import numpy as np
import os
import scipy
from scipy.sparse import *
from scipy.sparse.linalg import *
from sklearn import metrics
# from sklearn import preprocessing
from sklearn.cluster import KMeans
import random
from scipy.misc import comb
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.utils import np_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from esn_cell import ESNCell

def open_file(path_data, dataset, info):
    data_x = []
    data_y = []
    count = 0;
    for line in open(path_data):
        count = count + 1
        row = [[np.float32(x)] for x in line.strip().split(',')]
        label = np.int32(row[0])
        row = np.array(row[1:])
        row_shape = np.shape(row)
        row_mean = np.mean(row[0:])
        data_x.append(row - np.kron(np.ones((row_shape[0], row_shape[1])), row_mean))
        data_y.append(label[0])
    return data_x, data_y
def loading_ucr(index):
    dir_path = './UCR_TS_Archive_2015'
    list_dir = os.listdir(dir_path)
    dataset = list_dir[index]
    train_data = dir_path + '/' + dataset + '/' + dataset + '_TRAIN'
    test_data = dir_path + '/' + dataset + '/' + dataset + '_TEST'
    train_x, train_y = open_file(train_data, dataset, 'train')
    test_x, test_y = open_file(test_data, dataset, 'test')
    return train_x, train_y, test_x, test_y, dataset
def transfer_labels(labels_train, labels_test):
    indexes = np.unique(labels_train)
    num_classes = indexes.shape[0]
    num_samples_train = np.shape(labels_train)[0]
    num_samples_test = np.shape(labels_test)[0]
    for i in range(num_samples_train):
        new_label = np.argwhere(indexes == labels_train[i])[0][0]
        labels_train[i] = new_label
    labels_train = np_utils.to_categorical(labels_train, num_classes)
    for i in range(num_samples_test):
        new_label = np.argwhere(indexes == labels_test[i])[0][0]
        labels_test[i] = new_label
    labels_test = np_utils.to_categorical(labels_test, num_classes)
    return labels_train, labels_test, num_classes


def train_readout(M, T, n_res):
    lamb = 1e-2
    reservoir_units = n_res
    temp = tf.matmul(tf.transpose(M), T)
    readout = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(M), M) + lamb * tf.eye(reservoir_units)), temp)
    return readout


## juge measure
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


## Metric measure
def ED(x, Y,n):
    x_reshape_0 = tf.tile(tf.reshape(x, shape=[-1, 1, n]), (1, Y, 1))
    x_reshape_1 = tf.tile(tf.reshape(x, shape=[1, -1, n]), (Y, 1, 1))  #
    x_distance = x_reshape_0 - x_reshape_1
    distance = tf.reduce_sum(tf.square(x_distance), 2)
    return distance


def data_info():
    dir_path = './UCR_TS_Archive_2015'
    list_dir = os.listdir(dir_path)
    index = list_dir.index('Beef')
    train_x, train_y, test_x, test_y, dataset_name = loading_ucr(index=index)
    nb_train, len_series = np.shape(train_x)[0], np.shape(train_x)[1]
    nb_test = np.shape(test_x)[0]
    train_x = np.reshape(train_x, [nb_train, len_series])
    test_x = np.reshape(test_x, [nb_test, len_series])
    train_y, test_y, nb_class = transfer_labels(train_y,
                                                test_y)  # transfer label to nb*nb_class,if sample belong to the class,index is 1 else 0
    kmeans_o = KMeans(n_clusters=nb_class).fit(train_x)
    kmeans_orign = np_utils.to_categorical(kmeans_o.labels_, nb_class)  # as initial class label
    return train_x, train_y, nb_train, len_series, nb_class, kmeans_orign, dataset_name, test_x, test_y, nb_test


class ESN_clustering:
    def __init__(self,delta,lambda1,LK,epoch):
        # ----------------------config----------------------
        self.n_res =20 #the hidden size
        self.Connectivity = 1 ##connection rate
        self.LK = LK #leaking rate
        self.lambda1 = lambda1 
        self.learning_rate = 0.01 # learning rate
        self.nb_epoch = epoch #  Total training steps 
        self.delta =delta # RBF hyperparameter 
        self.lambda2=2

    def bulid_network(self):
        # -------------------bulid network------------------------
        ##----getting data--------
        train_x, train_y, nb_train, len_series, nb_class, kmeans_orign, filename, test_x, test_y, nb_test = data_info()
        time_step = len_series
        with tf.name_scope('input_layer'):
            X = tf.placeholder(tf.float32, [None, time_step, 1]) 
            Y = tf.placeholder(tf.int32)
            y_pseudo = tf.placeholder(tf.float32,[nb_class,nb_train])
            W=tf.placeholder(tf.float32,[self.n_res,nb_class])
        with tf.name_scope('ESN'):
            esn = ESNCell(num_units=self.n_res, connectivity=self.Connectivity, wr2_scale=1, leaky=self.LK)
            outputs, final_state = tf.nn.dynamic_rnn(esn, X, dtype=tf.float32)
            washed = tf.slice(outputs, [0, 0, 0], [-1, -1, -1])
    ##-----------------batch_size readout---------------------------
        wout_list = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        cond = lambda i, wout_list: tf.less(i, Y)
        def body(index, wout_list):
            readout = train_readout(washed[index, 0:-1, :], tf.reshape(X[index, 1:], (len_series - 1, 1)), self.n_res)
            wout_list = wout_list.write(index, tf.reshape(readout, (self.n_res,)))
            return index + 1, wout_list
        index = 0
        index, wout_list_final = tf.while_loop(cond, body, loop_vars=(index, wout_list))
        wout = wout_list_final.stack()
    ##------------------prediction loss------------------------------
        y_prediction = tf.matmul(washed[:, 0:-1, :], tf.reshape(wout, ([Y, self.n_res, 1])))
        wout=tf.transpose(wout)
        y_real = X[:,1:]
        pre_loss = tf.reduce_mean(tf.abs(y_prediction - y_real))
    ##-------------------------spectral_analysis---------------------
        with tf.name_scope('spectral_analysis'):
            distance = ED(wout,Y,self.n_res)
            matrix_G = tf.exp(-distance / (self.delta*self.delta))
            matrix_D = tf.matrix_diag(tf.reduce_sum(matrix_G, 0))
            matrix_L = matrix_D - matrix_G
            out_tmp = tf.matmul(tf.transpose(W),wout)
            cluster_loss = tf.norm(out_tmp - y_pseudo)
            cluster_result = tf.nn.softmax(y_pseudo)
            test_cluster_result = tf.nn.softmax(out_tmp)
        with tf.name_scope('train'):
            loss = pre_loss + self.lambda1 * cluster_loss
            win=tf.get_default_graph().get_tensor_by_name("rnn/ESNCell/InputMatrix:0")
            G1=tf.gradients(loss,win)
            optimizer_ESN = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss,
                                                                                              var_list=[var for var in
                                                                                                        tf.global_variables()
                                                                                                        if
                                                                                                    'ESN' in var.name])
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            model_path = './Model/model.ckpt'
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            W_reg=np.loadtxt('./Model/W_reg.txt')
            trian_T_label = np.argmax(train_y, 1)
            test_T_lable = np.argmax(test_y, 1)
            y_initial = (kmeans_orign.T).astype(np.float32)
            y_pse=y_initial
            batch_x_test = np.reshape(test_x, [nb_test, len_series, 1])
            test_cluster_result_ = sess.run([test_cluster_result], feed_dict={X: batch_x_test, Y: nb_test,y_pseudo: y_pse, W: W_reg})
            test_cluster_result_ = np.squeeze(np.array(test_cluster_result_), axis=0)
            test_P_lable = np.argmax(test_cluster_result_, 0)
            test_RI = rand_index_score(test_P_lable, test_T_lable)
        tf.reset_default_graph()
        return test_RI
if __name__=='__main__':
    ###-----------------------Hyperparameter  setting--------------------------------------
    lambda1=1
    LK =0.9
    delta=1
    epoch=15
    ###----------------------build networks-------------------------------------
    cluster=ESN_clustering(delta,lambda1,LK,epoch)
    RI=cluster.bulid_network()
    print('TEST_RI:',RI)