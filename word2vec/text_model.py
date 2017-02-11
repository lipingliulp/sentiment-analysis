from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
from scipy import sparse
import sys
from text_graph import GraphBuilder

def separate_valid(reviews, frac):
    review_size = len(reviews)
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)

    trainset = [reviews[i] for i in tind]
    validset = [reviews[j] for j in vind]
    
    return trainset, validset

# config = dict(K=K, voc_size=voc_size, use_sideinfo=use_sideinfo, half_window=half_window)
def fit_emb(reviews, config, init_model):

    use_valid_set = False
    if use_valid_set:
        reviews, valid_reviews = separate_valid(reviews, 0.1)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, init_model, training=True)

        optimizer = tf.train.AdagradOptimizer(1).minimize(outputs['objective'])
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_logg = np.zeros([config['max_iter'], 2]) 
        review_size = len(reviews)
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            words, context, preceding_tags = generate_batch(reviews, rind)
            feed_dict = {inputs['input_att']: preceding_tags, inputs['input_ind']: words, inputs['input_context']: context}

            _, llh_val, obj_val, debug_val = session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv']), feed_dict=feed_dict)
            loss_logg[step - 1, :] = np.array([llh_val, obj_val])

            # print loss every 2000 iterations
            nprint = 5000
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                
                valid_msg = ''
                if use_valid_set:
                    llh_sum = 0
                    valid_size = valid_reviews['scores'].shape[0]
                    for iv in xrange(valid_size): 
                        words, context, preceding_tags = generate_batch(reviews, rind)
                        feed_dict = {inputs['input_att']: preceding_tags, inputs['input_ind']: words, inputs['input_context']: context}
                        llh_val = session.run((outputs['llh']), feed_dict=feed_dict)
                        llh_sum = llh_sum + llh_val
                    valid_llh = llh_sum / valid_size
                    valid_msg = ' validation llh is %.3f' % valid_llh
                
                avg_val = np.mean(loss_logg[step - nprint : step], axis=0)
                print("iteration[", step, "]: average llh and obj are ", avg_val, valid_msg)
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    #debug_val = session.run(outputs['debugv'], feed_dict=feed_dict)
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
    
        # save model parameters to dict
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                     invmu=model_param['invmu'].eval(), 
                    weight=model_param['weight'].eval(), 
                       nbr=model_param['nbr'].eval())

        return model, loss_logg

def evaluate_emb(reviews, model, config):

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, config, model, training=False)
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_array = [] 
        pos_loss_array = [] 
        review_size = len(reviews)
        for step in xrange(review_size):
            words, context, preceding_tags = generate_batch(reviews, step)
            feed_dict = {inputs['input_att']: preceding_tags, inputs['input_ind']: words, inputs['input_context']: context}
            ins_llh_val = session.run((outputs['ins_llh']), feed_dict=feed_dict)
            loss_array.append(ins_llh_val)
          
        loss_array = np.array(loss_array) 
        print("Loss mean is ", np.mean(loss_array))
        # set a place holder. no use
        pos_loss_array = np.array([0])

        return loss_array, pos_loss_array

def generate_batch(reviews, rind):
    doc = reviews[rind].T
    text = doc[0, :]
    tags = doc[1, :]

    half_window = 1 
    span = 2 * half_window
    
    if text.shape[0] <= span:
        raise Exception('Document %d is too short' % rind)

    batch_size = len(text) - span

    label_ind = np.arange(half_window, batch_size + half_window)
    words = text[label_ind]

    context_ind = np.r_[np.arange(0, half_window), np.arange(half_window + 1, 2 * half_window + 1)]
    context_ind = np.tile(context_ind, [batch_size, 1]) + np.arange(0, batch_size)[:, None]
    context = text[context_ind] 

    preceding_tags = tags[label_ind - 1]

    return words, context, preceding_tags 

