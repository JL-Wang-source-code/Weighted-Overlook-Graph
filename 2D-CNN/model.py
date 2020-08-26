# -*- coding: utf-8 -*-

import tensorflow as tf

#%%
def inference1(images1, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    
    with tf.variable_scope('conv11') as scope:
        weights1 = tf.get_variable('weights1', 
                                  shape = [5,5,1,16],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases1 = tf.get_variable('biases1', 
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images1, weights1, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases1)
        conv11 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1 and norm1   
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.avg_pool(conv11, ksize=[1,5,5,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm1')
    
    #conv2----------------------------------------------------------------------------------
    with tf.variable_scope('conv12') as scope:
        weights1 = tf.get_variable('weights1',
                                  shape=[3,3,16,32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases1 = tf.get_variable('biases1',
                                 shape=[32], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights1, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases1)
        conv12 = tf.nn.relu(pre_activation, name='conv12')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.avg_pool(conv12, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm2')

    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights1 = tf.get_variable('weights1',
                                  shape=[dim,1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases1 = tf.get_variable('biases1',
                                 shape=[1024],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local13 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name=scope.name)    
    
    
    
    #local4-----------------------------------------------------------------------
    with tf.variable_scope('local4') as scope:
        weights1 = tf.get_variable('weights1',
                                  shape=[1024,512],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases1 = tf.get_variable('biases1',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local13, weights1) + biases1, name='local4')
    
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights1 = tf.get_variable('softmax_linear',
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases1 = tf.get_variable('biases1', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights1), biases1, name='softmax_linear')
    
    return softmax_linear

#%%
def losses1(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss1') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss1 = tf.reduce_mean(cross_entropy, name='loss1')
        tf.summary.scalar(scope.name+'/loss1', loss1)
    return loss1

#%%
def trainning1(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer1'):
        optimizer1 = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer1.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation1(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy1') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy1 = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy1', accuracy1)
  return accuracy1
