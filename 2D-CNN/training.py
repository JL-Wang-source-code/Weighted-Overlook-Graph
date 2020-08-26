# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import input_data
import model
import math
import time
#%%

N_CLASSES = 2
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 50
CAPACITY = 1500
MAX_STEP = 1500 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
n_test = 640
# you need to change the directories to yours.
train_dir1 = '/home/Bonntrain/'
test_dir1 = '/home/Bonntest/'

logs_train_dir1 = '/home/log/'
log_dir1 = logs_train_dir1

#     evaluate1()


#%%
def run_training():
    starttime1 = time.time()
    
    
    train1, train_label1 = input_data.get_files(train_dir1)
    
    train_batch1, train_label_batch1 = input_data.get_batch(train1,
                                                          train_label1,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)      
    train_logits = model.inference1(train_batch1, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses1(train_logits, train_label_batch1)        
    train_op = model.trainning1(train_loss, learning_rate)
    train__acc = model.evaluation1(train_logits, train_label_batch1)
       
    summary_op1 = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir1, sess.graph)
    saver1 = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op1)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir1, 'model1.ckpt')
                saver1.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    endtime1 = time.time()
    print (endtime1 - starttime1)





def evaluate1():
    with tf.Graph().as_default():
        

        prediction1all = np.array([],dtype = bool)
        
        # reading test data
        images1, labels = input_data.get_files1(test_dir1)
        
        test_batch1, test_label_batch = input_data.get_batch(images1,
                                                        labels,
                                                        IMG_W,
                                                        IMG_H,
                                                        BATCH_SIZE, 
                                                        CAPACITY)

        
        n_test = len(labels)
        logits1 = model.inference1(test_batch1, BATCH_SIZE, N_CLASSES)
        logits1 = tf.nn.softmax(logits1)
        pre = tf.argmax(logits1,1)
        top_k_op1 = tf.nn.in_top_k(logits1, test_label_batch, 1)
        

        saver1 = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess1:
            
            print("Reading checkpoints...")
            ckpt1 = tf.train.get_checkpoint_state(log_dir1)
            if ckpt1 and ckpt1.model_checkpoint_path:
                global_step = ckpt1.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver1.restore(sess1, ckpt1.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess1, coord = coord)
            
            try:
                num_iter = int(math.ceil(n_test // BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0
                
                pre_labels = []
                N_count = 0
                tp_count = 0
                fp_count = 0
                tn_count = 0
                fn_count = 0

                while step < num_iter and not coord.should_stop():
                    #predictions1 = sess1.run([top_k_op1])
                    
                    #pre_Y = sess1.run(pre)
                    pre_Y, predictions1, test_label_batch1 = sess1.run([pre, top_k_op1, test_label_batch])
                    pre_labels.extend(pre_Y)
                    
                    prediction1all = np.append(prediction1all,predictions1)
                    true_count += np.sum(predictions1)
                    step += 1
                    
                    for index in range(BATCH_SIZE):
                      if(test_label_batch1[index]==True):
                          N_count+=1
                      if(test_label_batch1[index]==True and pre_Y[index]==False):
                          fp_count+=1
                      if(test_label_batch1[index]==False and pre_Y[index]==True):
                          fn_count+=1
                      if(test_label_batch1[index]==False and pre_Y[index]==False):
                          tp_count+=1
                      if(test_label_batch1[index]==True and pre_Y[index]==True):
                          tn_count+=1    
                    
                    precision = true_count / total_sample_count
                    
                                                
                accuracy = (tp_count+tn_count)/(tn_count+fp_count+tp_count+fn_count)
                specificity = tn_count/(tn_count+fp_count)
                recall = tp_count/(tp_count+fn_count)
                
                print('fp_count = %.5f' % fp_count)
                print('fn_count = %.5f' % fn_count)
                print('tp_count = %.5f' % tp_count)
                print('tn_count = %.5f' % tn_count)
                
                print("precision = %.5f" % precision)    
                print('Acc = %.5f' % accuracy)
                print('Spec = %.5f' % specificity)
                print('Rec = %.5f' % recall)
                return prediction1all
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
                
                
run_training()
evaluate1()
