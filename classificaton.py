import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

minist = input_data.read_data_sets('MINIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size],name='W'))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1,)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={x:v_xs,y:v_ys})
    return result

#define palceholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784],name='x_in')#28*28
ys = tf.placeholder(tf.float32,[None,10],name='y_in')
#add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#loss
cross_entropy = tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
#training
for i in range(1000):
    batch_xs,batch_ys = minist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i %50 ==0:
        #to see the step improvement
        #print(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        #to visualize the result and improvement
        print(compute_accuracy(minist.test.images,minist.test.labels))
        #try:
            #ax.lines.remove(lines[0])
        #except Exception:
            #pass
        #prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        #plot the prediction
        #lines = ax.plot(x_data,prediction_value,'r-',lw =5)
        #plt.pause(0.1)

