from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf
import collections
from scipy import sparse
import sys
from graph_builder import GraphBuilder

def separate_valid(reviews, frac):
    review_size = reviews['scores'].shape[0]
    vind = np.random.choice(review_size, int(frac * review_size), replace=False)
    tind = np.delete(np.arange(review_size), vind)

    trainset = dict(scores=reviews['scores'][tind, :], atts=reviews['atts'][tind])
    validset = dict(scores=reviews['scores'][vind, :], atts=reviews['atts'][vind])
    
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

        optimizer = tf.train.AdagradOptimizer(0.1).minimize(outputs['objective'])
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_logg = np.zeros([config['max_iter'], 2]) 
        review_size = reviews['scores'].shape[0]
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            atts, indices, labels = generate_batch(reviews, rind)
            feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}

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
                        atts, indices, labels = generate_batch(valid_reviews, iv)
                        feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}
                        llh_val = session.run((outputs['llh']), feed_dict=feed_dict)
                        llh_sum = llh_sum + llh_val
                    valid_llh = llh_sum / valid_size
                    valid_msg = ' validation llh is %.3f' % valid_llh
                
                avg_val = np.mean(loss_logg[step - nprint : step], axis=0)
                print("iteration[", step, "]: average llh and obj are ", avg_val, valid_msg)
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    debug_val = session.run(outputs['debugv'], feed_dict=feed_dict)
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
    
        # save model parameters to dict
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                     invmu=model_param['invmu'].eval(), 
                    weight=model_param['weight'].eval())

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
        review_size = reviews['scores'].shape[0]
        for step in xrange(review_size):
            att, index, label = generate_batch(reviews, step)
            feed_dict = {inputs['input_att']: att, inputs['input_ind']: index, inputs['input_label']: label}
            ins_llh_val, pos_llh_val = session.run((outputs['ins_llh'], outputs['pos_llh']), feed_dict=feed_dict)
            loss_array.append(ins_llh_val)
            pos_loss_array.append(pos_llh_val)
        
        print("Loss and pos_loss mean: ", np.mean(np.concatenate(loss_array)), np.mean(np.concatenate(pos_loss_array)))
        
        return loss_array, pos_loss_array

def generate_batch(reviews, rind):
    atts = reviews['atts'][rind, :]
    _, ind, rate = sparse.find(reviews['scores'][rind, :])
    return atts, ind, rate 

