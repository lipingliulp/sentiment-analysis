'''run restaurant experiment'''
import sys
import string
#from  restaurant_experiment import restaurant_experiment
from  movie_experiment import movie_experiment


def parse_value(strv):
    if strv == 'False':
        value = False 
    elif strv == 'True':
        value = True
    elif '.' in strv:
        value = float(strv)
    else: 
        value = int(strv)
    
    return value

if __name__ == '__main__':

    #config = dict(use_sideinfo=False, K=16, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=True) 
    dataset = 'movie'
    max_iter = 200000
    dist = 'binomial'

    #dataset = 'market'
    #max_iter = 400000
    #dist = 'poisson'

    config = dict(K=16, exposure=True, use_sideinfo=True, dist=dist, zeroweight=1.0,
               max_iter=max_iter, ar_sigma2=1, w_sigma2=1, cont_train=False, sample_ratio=0.1, fold=0) 

    for iarg in xrange(1, len(sys.argv)):
        arg = sys.argv[iarg]
        key, strv = string.split(arg, '=')
        
        value = parse_value(strv)
        
        if key in config:
            config[key] = value
        else:
            raise Exception(key + ' is not a key of the config')

    print('The configuration is: ')
    print(config)

    #restaurant_experiment(config)
    movie_experiment(config, dataset)

