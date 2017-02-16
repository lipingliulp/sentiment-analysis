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

def calculate_similarity(embeddings, valid_dataset):
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    
    return similarity


# config = dict(K=K, voc_size=voc_size, use_sideinfo=use_sideinfo, half_window=half_window)
def fit_emb(reviews, config, init_model, reverse_dictionary):

    use_valid_set = True 
    if use_valid_set:
        reviews, valid_reviews = separate_valid(reviews, 0.1)

    check_size = 20
    check_words = np.random.choice(len(reverse_dictionary), check_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, reverse_dictionary, config, init_model, training=True)
        similarity = calculate_similarity(model_param['rho'], tf.constant(check_words)) 

        optimizer = tf.train.AdagradOptimizer(1).minimize(outputs['objective'])
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        nprint = 10000
        val_accum = np.array([0.0, 0.0])
        loss_logg = np.zeros([int(config['max_iter'] / nprint) + 1, 3]) 

        review_size = len(reviews)
        for step in xrange(1, config['max_iter'] + 1):

            rind = np.random.choice(review_size)
            words, context, preceding_tags = generate_batch(reviews, rind)
            feed_dict = {inputs['input_att']: preceding_tags, inputs['input_ind']: words, inputs['input_context']: context}

            _, llh_val, obj_val, debug_val = session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv']), feed_dict=feed_dict)
            val_accum = val_accum + np.array([llh_val, obj_val])
            
            # print loss every nprint iterations
            if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
                
                valid_llh = 0.0
                if use_valid_set:
                    llh_sum = 0.0
                    valid_size = len(valid_reviews)
                    for iv in xrange(valid_size): 
                        words, context, preceding_tags = generate_batch(valid_reviews, iv)
                        feed_dict = {inputs['input_att']: preceding_tags, inputs['input_ind']: words, inputs['input_context']: context}
                        llh_val = session.run((outputs['llh']), feed_dict=feed_dict)
                        llh_sum = llh_sum + llh_val
                    valid_llh = llh_sum / valid_size
                
                # record the three values 
                ibatch = int(step / nprint)
                loss_logg[ibatch, :] = np.append(val_accum / nprint, valid_llh)
                val_accum[:] = 0.0 # reset the accumulater

                print("iteration[", step, "]: average llh, obj and debugv are ", loss_logg[ibatch, :])
                
                if np.isnan(llh_val) or np.isinf(llh_val):
                    #debug_val = session.run(outputs['debugv'], feed_dict=feed_dict)
                    print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
                    raise Exception('Bad values')
    
            if step % 50000 == 0: 
                print('----------------------------------------------------------------------')
                sim = similarity.eval()
                for i in xrange(check_size):
                    word = reverse_dictionary[check_words[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)

                    print(log_str)
                
        # save model parameters to dict
        model = dict(alpha=model_param['alpha'].eval(), 
                       rho=model_param['rho'].eval(), 
                     invmu=model_param['invmu'].eval(), 
                    weight=model_param['weight'].eval(), 
                       nbr=model_param['nbr'].eval())

        return model, loss_logg

def evaluate_emb(reviews, model, config, reverse_dictionary):

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(27)
        # construct model graph
        builder = GraphBuilder()
        inputs, outputs, model_param = builder.construct_model_graph(reviews, reverse_dictionary, config, model, training=False)
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()

        loss_array = [] 
        pos_loss_array = [] 
        review_size = len(reviews)
        print('%d documents for testing...')
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

    # each word is paired with the tag of its preceding word
    preceding_tags = tags[label_ind]

    return words, context, preceding_tags 

