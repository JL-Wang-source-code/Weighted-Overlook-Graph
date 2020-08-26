# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os



def get_files(file_dir1):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    zero = []
    label_zero = []
    one = []
    label_one = []

    for file1 in os.listdir(file_dir1):
        name = file1.split(sep='_')
        if name[0]=='0':
            zero.append(file_dir1 + file1)
            label_zero.append(0)
        else:
            one.append(file_dir1 + file1)
            label_one.append(1)
    
    image1_list = np.hstack((zero, one))
    label_list = np.hstack((label_zero, label_one))
    
    temp = np.array([image1_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image1_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    
    return image1_list, label_list


def get_batch(image1, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image1 = tf.cast(image1, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image1, label])
    
    label = input_queue[1]
    image1_contents = tf.read_file(input_queue[0])
    image1 = tf.image.decode_png(contents=image1_contents, channels=1)
    
    image1 = tf.image.resize_image_with_crop_or_pad(image1, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.
    image1 = tf.image.per_image_standardization(image1)
    
    image1_batch, label_batch = tf.train.batch([image1, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image1_batch = tf.cast(image1_batch, tf.float32)
    
    return image1_batch, label_batch


def get_files1(file_dir1):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    zero = []
    label_zero = []
    one = []
    label_one = []

    for file1 in os.listdir(file_dir1):
        name = file1.split(sep='_')
        if name[0]=='0':
            zero.append(file_dir1 + file1)
            label_zero.append(0)
        else:
            one.append(file_dir1 + file1)
            label_one.append(1)
    
    image1_list = np.hstack((zero, one))
    label_list = np.hstack((label_zero, label_one))
    
    temp = np.array([image1_list, label_list])
    temp = temp.transpose()
    #np.random.shuffle(temp)
    
    image1_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    
    return image1_list, label_list
