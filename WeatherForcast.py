__author__ = 'ahmaddorri'
from sklearn import preprocessing

import pandas as pd
import numpy as np
import tensorflow as tf
import random as rd
#with tf.device('/gpu:0'):
xls = pd.ExcelFile("TehranWeather2008.xls")
sheet = xls.parse(0)

sheet = sheet.dropna(subset=['T'])
sheet = sheet.drop(['#    Record fields order: date', '# time', 'db_id', 'C'], axis=1)
sheet = sheet.interpolate()

sheet_train = pd.DataFrame(sheet.values[:800].astype(float))
sheet_train_normal = (sheet_train - sheet_train.mean()) / (sheet_train.max() - sheet_train.min())
sheet_train_normal = pd.DataFrame(sheet_train_normal)

sheet_test = pd.DataFrame(sheet.values[800:].astype(float))
sheet_test_normal = (sheet_test - sheet_train.mean()) / (sheet_train.max() - sheet_train.min())
sheet_test_normal = pd.DataFrame(sheet_test_normal)

#print(sheet_test_normal)
x_train = []
y_train = []

for i in range(0, 795):
    s = np.asarray(sheet_train_normal[i:i+6])
    s1 = np.asarray(sheet_train[i:i+6])
    ap = np.append(arr=[s[0], s[1], s[2], s[3], s[4]], values=[])
    bp = np.append(arr=s[5][:10],values=s[5][11:])
    cp = np.append(arr=ap, values=bp)
    x_train.append(cp)
    y_train.append([s1[5][10]])

x_train = np.asanyarray(x_train)
x_train = x_train.tolist()

y_train = np.asanyarray(y_train)
y_train = y_train.tolist()

#min_max_scaler_x = preprocessing.MinMaxScaler()
#x_train = min_max_scaler_x.fit_transform(x_train)

#min_max_scaler_y = preprocessing.MinMaxScaler()
#y_train = min_max_scaler_y.fit_transform(y_train)


x_test = []
y_test = []
for i in range(0, 390):
    s = np.asarray(sheet_test_normal[i:i+6])
    s1 = np.asarray(sheet_test[i:i+6])
    ap = np.append(arr=[s[0], s[1], s[2], s[3], s[4]], values=[])
    bp = np.append(arr=s[5][:10],values=s[5][11:])
    cp = np.append(arr=ap, values=bp)
    x_test.append(cp)
    y_test.append([s1[5][10]])
x_test = np.asanyarray(x_test)
x_test = x_test.tolist()

y_test = np.asanyarray(y_test)
y_test = y_test.tolist()
#x_test=min_max_scaler_x.transform(x_test)
#y_test=min_max_scaler_y.transform(y_test)

x_t = []
y_t = []
for i in range(0, 390):
    s = np.asarray(sheet_test[i:i+6])
    ap = np.append(arr=[s[0], s[1], s[2], s[3], s[4]], values=[])
    bp = np.append(arr=s[5][:10],values=s[5][11:])
    cp = np.append(arr=ap, values=bp)
    x_t.append(cp)
    y_t.append([s[5][10]])
x_t = np.asanyarray(x_t)
x_t = x_t.tolist()

y_t = np.asanyarray(y_t)
y_t = y_t.tolist()

h_layer_1 = input("how many neuron in hidden layer 1 :")
h_layer_1 = int(h_layer_1)

h_layer_2 = input("how many neuron in hidden layer 2 :")
h_layer_2 = int(h_layer_2)

x_ = tf.placeholder(tf.float32, shape=[None, 83], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

Theta1 = tf.Variable(tf.truncated_normal([83, h_layer_1]), name="Theta1")
Theta2 = tf.Variable(tf.truncated_normal([h_layer_1, h_layer_2]), name="Theta2")
Theta3 = tf.Variable(tf.truncated_normal([h_layer_2, 1]), name="Theta3")

Bias1 = tf.Variable(tf.zeros([h_layer_1]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([h_layer_2]), name="Bias2")
Bias3 = tf.Variable(tf.zeros([1]), name="Bias3")

with tf.name_scope("layer2") as scope:
    A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

with tf.name_scope("layer3") as scope:
    A3 = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

with tf.name_scope("layer4") as scope:
    Hypothesis = (tf.matmul(A3, Theta3))+Bias3

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.squared_difference(Hypothesis, y_))

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

th = input("threshold for stop?")
th =float(th)
print("start train ...")
for i in range(100000):
    sess.run(train_step, feed_dict={x_: x_train, y_: y_train})
    print("_"*50)
    print('Epoch ', i)
    c = sess.run(cost, feed_dict={x_: x_train, y_: y_train})
    print('cost ', c)
    if(c<th):
        print("in itr ", i, " converge")
        break

#print("train finish ")
#print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: x_train, y_: y_train}))
#print('Theta1 ', sess.run(Theta1))
#print('Bias1 ', sess.run(Bias1))
#print('Theta2 ', sess.run(Theta2))
#print('Bias2 ', sess.run(Bias2))
#print('Theta3 ', sess.run(Theta3))
#print('Bias3 ', sess.run(Bias3))
#print('cost ', sess.run(cost, feed_dict={x_: x_train, y_: y_train}))


print("-"*80)
print("MSE for test is :")
print(sess.run(cost, feed_dict={x_: x_test, y_: y_test}))

h=sess.run(Hypothesis, feed_dict={x_: x_test, y_: y_test})
print(h)
