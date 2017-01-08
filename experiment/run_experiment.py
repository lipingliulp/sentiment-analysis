'''run restaurant experiment'''
import sys
import string
#from  restaurant_experiment import restaurant_experiment
from  movie_experiment import movie_experiment

if __name__ == '__main__':

    #config = dict(use_sideinfo=False, K=16, max_iter=800000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=True) 
    config = dict(use_sideinfo=True, K=32, max_iter=100000, reg_weight=0.01, exposure=True, cont_train=False, sample_ratio=0.4)
   
    for iarg in xrange(1, len(sys.argv)):
        arg = sys.argv[iarg]
        key, strv = string.split(arg, '=')

        value = False if strv == 'False' else (True if strv == 'True' else int(strv)) 
        
        if key in config:
            config[key] = value
        else:
            raise Exception(key + ' is not a key of the config')
   
    print('The configuration is: ')
    print(config)

    #restaurant_experiment(config)
    movie_experiment(config)

