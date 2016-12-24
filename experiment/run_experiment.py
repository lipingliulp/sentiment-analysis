'''run restaurant experiment'''
import sys
import string
from  restaurant_experiment import restaurant_experiment

if __name__ == '__main__':

    config = dict(use_sideinfo=False, K=64, max_iter=500000, half_window=1, reg_weight=0.01, num_neg=500, negpos_ratio=1000, exposure=False, cont_train=True)
    
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

    restaurant_experiment(config)

