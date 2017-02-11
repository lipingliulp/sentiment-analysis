
import tensorflow as tf
import numpy as np

class GraphBuilder:
    def __init__(self): 
        
        self.alpha = None
        self.rho = None

        self.invmu = None
        self.weight = None

        self.nbr = [] 

        self.input_att = None
        self.input_ind = None
        self.input_label = None
        self.context = None

        self.cdist = None

    def logprob_nonz(self, alpha_emb, config, training=True):

        # for word embedding, rate is 1
        #rate = tf.cast(self.input_label, tf.float32)
        rho_select = tf.gather(self.rho, self.input_ind)
        # binomial distribution
        emb = tf.reduce_sum(rho_select * alpha_emb, reduction_indices=1)

        # only bernoulli distribution
        logplusprob = - tf.nn.softplus(- emb)
        logprob_nz = logplusprob 

        # also calculate the probability of getting zeros, will be substracted from the noisy estimation of the gradient
        logminusprob = - tf.nn.softplus(emb)
   
        if config['exposure']:
            logits = tf.gather(self.invmu, self.input_ind)
            if config['use_sideinfo']:
                # input feature is one-hot vector, and the non-zero index is recorded in input_att
                # weight select
                weight_feat = tf.gather_nd(self.weight, tf.pack([self.input_ind, self.input_att], axis=1))
                logits = logits + weight_feat 
            
            log_obs_prob = - tf.nn.softplus(- logits) 
            logprob = log_obs_prob + logprob_nz

            # calculate the probability of getting zeros with the exposure model
            log_nobs_prob = - tf.nn.softplus(logits)
            logprob_zero = self.logsumexp(log_obs_prob + logminusprob, log_nobs_prob)

        else:
            logprob = logprob_nz
            # calculate the probability of getting zeros without the exposure model
            logprob_zero = logminusprob
        
        return logprob, logprob_nz, logprob_zero, []


    def logprob_zero(self, context_emb, config, training):
        
        # get index of zeros
        voc_size = int(self.rho.get_shape()[0])

        # random sample words for all pairs,  to estimate sum_i sum_w log p(b_iw | c_i)
        # subsample positions: sum_i' sum_w log p(b_i'w | c_i) = sum_w sum_i' log p(b_i'w | c_i)
        # subsample words: sum_w' sum_i' log p(b_i'w' | c_i) 
       
        nsample = tf.cast(config['sample_ratio'] * voc_size, tf.int32)
        sind = self.cdist.sample(sample_shape=nsample)

        rho_z = tf.gather(self.rho, sind) 
        emb = tf.matmul(context_emb, rho_z, transpose_b=True)

        # only bernoulli distribution
        logprob_z  = - tf.nn.softplus(emb)

        if config['exposure']:
            logits = tf.gather(self.invmu, sind)
            tf.expand_dims(logits, axis=0)
            if config['use_sideinfo']:
                weight_select = tf.transpose(tf.gather(self.weight, sind))
                weight_feat = tf.gather(weight_select, self.input_att)
                logits = logits + weight_feat 
                #logits = logits + tf.reduce_sum(weight_z * tf.expand_dims(input_att, 0), 1)
            
            log_nobs_prob = - tf.nn.softplus(logits)
            log_obs_prob = - tf.nn.softplus(- logits) 
            logprob = self.logsumexp(log_obs_prob + logprob_z, log_nobs_prob)
    
        else:
            logprob = logprob_z
    
        return logprob, sind, []

    def construct_model_graph(self, reviews, config, init_model=None, training=True):

        review_size, voc_size, dim_atts = self.get_problem_sizes(reviews, config)
        self.initialize_model(review_size, voc_size, dim_atts, config, init_model, training)

        # number pairs in the batch
        batch_size = tf.cast(tf.shape(self.input_ind)[0], tf.float32)
        
        ###### construct alpha_sum
        alpha_select = tf.gather(self.alpha, self.input_context)
        alpha_sum = tf.reduce_sum(alpha_select, reduction_indices=1)

        llh_nonz, emb_logp, llh_fake_zero, _ = self.logprob_nonz(alpha_sum, config=config, training=training)
        sum_llh_nonz = tf.reduce_sum(llh_nonz)
        sum_llh_fake_zero = tf.reduce_sum(llh_nonz)

        llh_zero, sind, debug = self.logprob_zero(alpha_sum, config=config, training=training)
        mean_llh_zero = tf.reduce_mean(llh_zero)
        

        ## combine logprob of single instances
        #if not training:
        #    ins_logprob = tf.concat(0, [llh_zero, llh_nonz])
        #    ins_ind = tf.concat(0, [sind, self.input_ind])
        #    ins_llh = tf.scatter_update(tf.Variable(tf.zeros(voc_size)), ins_ind, ins_logprob)
        #    sum_llh = tf.reduce_sum(llh_nonz) + tf.reduce_sum(llh_zero) 
        #else:
        sum_llh = sum_llh_nonz + (mean_llh_zero * batch_size * float(voc_size) - sum_llh_fake_zero) / 1000
        ins_llh = sum_llh 
        #raise  Exception('Do not use "training" option for text')
    
        # random choose weight vectors to get a noisy estimation of the regularization term
        rsize = int(voc_size * 0.005)
        rind = self.cdist.sample(sample_shape=rsize)
        regularizer = (tf.reduce_sum(tf.square(tf.gather(self.rho,   rind)))  \
                     + tf.reduce_sum(tf.square(tf.gather(self.alpha, rind)))) \
                      * (0.5 * voc_size / (config['ar_sigma2'] * rsize * review_size))
                    # (0.5 / sigma2): from Gaussian prior
                    # (voc_size / rsize): estimate the sum of squares of ALL vectors
                    # / review_size: the overall objective is scaled down by review size
        
        if config['use_sideinfo']:
            wreg = tf.reduce_sum(tf.square(tf.gather(self.weight, rind))) \
                   * (0.5 * voc_size / (config['w_sigma2'] * rsize * review_size))
            regularizer = regularizer + wreg
    
        objective = regularizer  - sum_llh  # the objective is an estimation of the llh of data divied by review_size
    
        inputs = {'input_att': self.input_att, 'input_ind': self.input_ind, 'input_context': self.input_context} 
        outputs = {'objective': objective, 'llh': sum_llh, 'ins_llh': ins_llh, 'pos_llh': emb_logp, 'debugv': debug}
        model_param = {'alpha': self.alpha, 'rho': self.rho, 'weight': self.weight, 'invmu': self.invmu, 'nbr': self.nbr}
    
        return inputs, outputs, model_param 
    

    def initialize_model(self, review_size, voc_size, dim_atts, config, init_model=None, training=True):

        embedding_size = config['K']
        # all of these variables will be used as indices 
        self.input_context = tf.placeholder(tf.int32, shape=[None, 2])
        self.input_att = tf.placeholder(tf.int32, shape=[None])
        self.input_ind = tf.placeholder(tf.int32, shape=[None])
        #self.input_label = tf.placeholder(tf.int32, shape=[None])
        # do not use nbr which is for negative binomial distribution 
        self.cdist = tf.contrib.distributions.Categorical(logits=np.zeros(voc_size))

        if training: 
            if init_model == None:
                self.weight = tf.Variable(tf.zeros([voc_size, dim_atts]))
                self.alpha  = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1, 1))
                self.rho    = tf.Variable(tf.truncated_normal([voc_size, embedding_size],stddev=1))
                self.invmu  = tf.Variable(tf.random_uniform([voc_size], -1, 1))
                self.nbr  = tf.Variable([])
            else:
                self.alpha  = tf.Variable(init_model['alpha'])
                self.invmu  = tf.Variable(init_model['invmu'])
                self.rho    = tf.Variable(init_model['rho'])
                self.weight = tf.Variable(init_model['weight'])

                free_nbr = self.inv_softplus_np(init_model['nbr'])
                self.nbr  = tf.nn.softplus(tf.Variable(free_nbr))
                print('use parameters of the initial model')
        else: 
            self.alpha  = tf.constant(init_model['alpha'])
            self.invmu  = tf.constant(init_model['invmu'])
            self.rho    = tf.constant(init_model['rho'])
            self.weight = tf.constant(init_model['weight'])
            self.nbr = tf.constant(init_model['nbr'])
 

    def get_problem_sizes(self, reviews, config):
        review_size = len(reviews) 
        voc_size = 20000 
        dim_atts = 12
        
        return review_size, voc_size, dim_atts

    def logsumexp(self, vec1, vec2):
        flag = tf.greater(vec1, vec2)
        maxv = tf.select(flag, vec1, vec2)
        lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
        return lse

    def gammaln(self, x):
        # fast approximate gammaln from paul mineiro
        # http://www.machinedlearnings.com/2011/06/faster-lda.html
        logterm = tf.log (x * (1.0 + x) * (2.0 + x))
        xp3 = 3.0 + x
        return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)


    def inv_softplus_np(self, x):
        y = np.log(np.exp(x) - 1)
        return y 
   
