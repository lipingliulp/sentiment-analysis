
import tensorflow as tf
import numpy as np

class GraphBuilder:
    def __init__(self): 
        
        self.alpha = None
        self.rho = None

        self.invmu = None
        self.weight = None

        self.nbr = None


        self.input_att = None
        self.input_ind = None
        self.input_label = None

    def logprob_nonz(alpha_emb, config, training=True):

        rate = tf.cast(self.input_label, tf.float32)
        rho_select = tf.gather(self.rho, self.input_ind)
        weight_select = tf.gather(self.weight, self.input_ind)
        # binomial distribution
        emb = tf.reduce_sum(rho_select * alpha_emb, reduction_indices=1)
        if config['dist'] == 'binomial':
            logminusprob = - tf.nn.softplus(emb)
            logplusprob = - tf.nn.softplus(- emb)
            logprob_nz = np.log(6.0) - self.gammaln(rate + 1.0) - self.gammaln(4.0 - rate) + rate * logplusprob + (3.0 - rate) * logminusprob

        elif config['dist'] == 'poisson':
            lamb = tf.nn.softplus(emb) + 1e-6
            logprob_nz = - self.gammaln(rate + 1.0) + rate * tf.log(lamb) - lamb 

        elif config['dist'] == 'negbin':
            nbr_select = tf.gather(self.nbr, self.input_ind)
            mu = tf.nn.softplus(emb) + 1e-6
            logprob_nz = self.gammaln(rate + nbr_select) - self.gammaln(rate + 1.0) -  self.gammaln(nbr_select) + \
                         nbr_select * tf.log(nbr_select) + rate * tf.log(mu) - (nbr_select + rate) * tf.log(nbr_select + mu)
        else:
            raise Exception('The distribution "' + config['dist'] + '" is not defined in the model')
             
        if config['exposure']:
            logits = tf.gather(self.invmu, self.input_ind)
            if config['use_sideinfo']:
                logits = logits + tf.reduce_sum(weight_select * tf.expand_dims(self.input_att, 0), 1)
            
            log_obs_prob = - tf.nn.softplus(- logits) 
            logprob = log_obs_prob + logprob_nz
        else:
            logprob = logprob_nz
        
        logprob_sum = tf.reduce_sum(logprob)
        
        return logprob, logprob_nz, []


    def logprob_zero(context_emb, config, training):
        
        # get index of zeros
        movie_size = int(self.rho.get_shape()[0])
        # TF allocates the variable once
        flag = tf.Variable(tf.ones(movie_size, dtype=tf.bool))
        # TF update the variable for every mini-batch 
        flag = tf.scatter_update(flag, np.arange(movie_size), np.tile([True], movie_size))
        flag = tf.scatter_update(flag, self.input_ind, tf.tile([False], tf.shape(self.input_ind)))
        sind = tf.cast(tf.squeeze(tf.where(flag)), tf.int32)
        if training: # if training, then subsample sind
            nsample = tf.cast(config['sample_ratio'] * tf.cast(tf.shape(sind)[0], dtype=tf.float32), tf.int32)
            sind = tf.gather(tf.random_shuffle(sind), tf.range(nsample))
  
        rho_z = tf.gather(self.rho, sind) 
        weight_z = tf.gather(self.weight, sind)

        emb = tf.reduce_sum(rho_z * context_emb, reduction_indices=1)
        if config['dist'] == 'binomial':
            # binomial distribution
            # p := tf.sigmoid(emb) 
            # log(1 - p)  := - tf.nn.softplus(emb)
            logprob_z  = - 3.0 * tf.nn.softplus(emb)

        elif config['dist'] == 'poisson':
            # poisson distribution
            lamb_z = tf.nn.softplus(emb) + 1e-6
            logprob_z = - lamb_z 

        elif config['dist'] == 'negbin':
            nbr_z = tf.gather(self.nbr, sind)
            mu = tf.nn.softplus(emb) + 1e-6
            logprob_z = nbr_z * tf.log(nbr_z) - nbr_z * tf.log(nbr_z + mu)

        else:
            raise Exception('The distribution "' + config['dist'] + '" is not defined in the model')
    
        if config['exposure']:
            logits = tf.gather(self.invmu, sind)
            if config['use_sideinfo']:
                logits = logits + tf.reduce_sum(weight_z * tf.expand_dims(input_att, 0), 1)
            
            log_nobs_prob = - tf.nn.softplus(logits) 
            log_obs_prob = - tf.nn.softplus(-logits) 
            logprob = self.logsumexp(log_obs_prob + logprob_z, log_nobs_prob)
    
        else:
            logprob = logprob_z
    
        if training:
            noisy_logprob_sum = tf.reduce_sum(logprob) * tf.cast(movie_size, tf.float32) / tf.cast(tf.shape(sind)[0], tf.float32)
        else:
            noisy_logprob_sum = tf.reduce_sum(logprob)
        
        return noisy_logprob_sum, logprob, sind, [tf.reduce_min(logprob_z), tf.reduce_max(logprob_z)]


    def initialize_model(review_size, movie_size, dim_atts, config, init_model=None, training=True):

        self.input_att = tf.placeholder(tf.float32, shape=[dim_atts])
        self.input_ind = tf.placeholder(tf.int32, shape=[None])
        self.input_label = tf.placeholder(tf.int32, shape=[None])
    
        if training: 
            if init_model == None:
                self.weight = tf.Variable(tf.zeros([movie_size, dim_atts]))
                self.alpha  = tf.Variable(tf.random_uniform([movie_size, embedding_size], -1, 1))
                self.rho    = tf.Variable(tf.truncated_normal([movie_size, embedding_size],stddev=1))
                self.invmu  = tf.Variable(tf.random_uniform([movie_size], -1, 1))
            else:
                self.alpha  = tf.Variable(init_model['alpha'])
                self.invmu  = tf.Variable(init_model['invmu'])
                self.rho    = tf.Variable(init_model['rho'])
                self.weight = tf.Variable(init_model['weight'])
                print('use parameters of the initial model')
        else: 
            self.alpha  = tf.constant(init_model['alpha'])
            self.invmu  = tf.constant(init_model['invmu'])
            self.rho    = tf.constant(init_model['rho'])
            self.weight = tf.constant(init_model['weight'])
 


    def construct_model_graph(reviews, config, init_model=None, training=True):

        embedding_size = config['K']
        review_size, movie_size, dim_atts = self.get_problem_sizes(reviews, config)
        self.initialize_model(review_size, movie_size, dim_atts, config, init_model, training)

        # rate 
        nnz = tf.cast(tf.shape(rate)[0], tf.float32)
    
        #prepare embedding of context 
        alpha_select = tf.gather(self.alpha, self.input_ind, name='context_alpha')
        alpha_weighted = alpha_select * tf.expand_dims(rate, 1)
        alpha_sum = tf.reduce_sum(alpha_weighted, keep_dims=True, reduction_indices=0)

        asum_zero = alpha_sum / nnz 
        asum_nonz = (alpha_sum - alpha_select) / (nnz - 1) 
    
        llh_zero, sind, _ = self.logprob_zero(asum_zero, config=config, training=training)
        llh_nonz, emb_logp, debug = self.logprob_nonz(asum_nonz, config=config, training=training)
        
        # combine logprob of single instances
        if not training:
            ins_logprob = tf.concat(0, [z_insprob, nz_insprob])
            ins_ind = tf.concat(0, [sind, input_ind])
            ins_llh = tf.scatter_update(tf.Variable(tf.zeros(movie_size)), ins_ind, ins_logprob)
            sum_llh = tf.reduce_sum(llh_nonz) + tf.reduce_sum(llh_zero) 
        else:
            ins_llh = None
            sum_llh = tf.reduce_sum(llh_nonz) + tf.reduce_mean(llh_zero) * (float(movie_size) - nnz)
   
        
        # random choose weight vectors to get a noisy estimation of the regularization term
        rsize = int(movie_size * 0.1)
        rind = tf.random_shuffle(np.arange(movie_size))[0 : rsize]
        regularizer = (tf.reduce_sum(tf.square(tf.gather(rho,   rind)))  \
                     + tf.reduce_sum(tf.square(tf.gather(alpha, rind)))) \
                      * (0.5 * movie_size / (config['ar_sigma2'] * rsize * review_size))
                    # (0.5 / sigma2): from Gaussian prior
                    # (movie_size / rsize): estimate the sum of squares of ALL vectors
                    # / review_size: the overall objective is scaled down by review size
        
        if config['use_sideinfo']:
            wreg = tf.reduce_sum(tf.square(tf.gather(self.weight, rind))) \
                   * (0.5 * movie_size / (config['w_sigma2'] * rsize * review_size))
            regularizer = regularizer + wreg
    
        objective = regularizer  - sum_llh  # the objective is an estimation of the llh of data divied by review_size
    
        inputs = {'input_att': self.input_att, 'input_ind': self.input_ind, 'input_label': self.input_label} 
        outputs = {'objective': objective, 'llh': sum_llh, 'ins_llh': ins_llh, 'pos_llh': emb_logp, 'debugv': debug}
        model_param = {'alpha': self.alpha, 'rho': self.rho, 'weight': self.weight, 'invmu': self.invmu, 'nbr', self.nbr}
    
        return inputs, outputs, model_param 
    

    def get_problem_sizes(reviews, config):
        review_size = reviews['scores'].shape[0]
        movie_size = reviews['scores'].shape[1]
        dim_atts = reviews['atts'].shape[1]
        
        return review_size, movie_size, dim_atts

    def logsumexp(vec1, vec2):
        flag = tf.greater(vec1, vec2)
        maxv = tf.select(flag, vec1, vec2)
        lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
        return lse

     def gammaln(x):
        # fast approximate gammaln from paul mineiro
        # http://www.machinedlearnings.com/2011/06/faster-lda.html
        logterm = tf.log (x * (1.0 + x) * (2.0 + x))
        xp3 = 3.0 + x
        return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)



   
