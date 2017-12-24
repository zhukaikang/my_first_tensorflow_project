import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist. import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size],name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name = 'b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

x_data = np.linspace(-1,1,300,dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_in')
    ys = tf.placeholder(tf.float32,[None,1],name='y_in')

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('tain'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#training
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i %50 ==0:
        #to see the step improvement
        #print(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        #to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        #plot the prediction
        lines = ax.plot(x_data,prediction_value,'r-',lw =5)
        plt.pause(0.1)

