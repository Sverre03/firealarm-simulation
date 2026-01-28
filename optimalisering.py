import numpy as np
import scipy.stats

def bayes(observations: np.ndarray, responses: np.ndarray, family):
    ''' 
    observations: nxm, 
    responses: nx1, coverage measure
    n: different simulations, 
    m: number of alarms
    '''

    posterior = np.exp()
    return np.argmax(posterior)
    
def loop(x0, sim_func: function, acq_func: function, sigma_e):
    #https://en.wikipedia.org/wiki/Bayesian_optimization#Acquisition_functions
    
    #x: grid
    #score = sim_func(x0)
    #current_acq = acq_func(x0,score) 
    #acq_max = Inf

    #while acq_max > tol:
        #x_next =  x[argmax(current_acq)]
        #score = sim_func(x_next)
    
        #current_acq = acq_func(x_next,score)
        #acq_max = max(current_acq_func)


    #need to apply symmetry of fire alarms: (x1,x2,x3) == (x2,x1,x3) osv.
    return

def expected_improvement(mu_x, sigma_x, f_xplus, xi=0):
    Z = (mu_x - f_xplus - xi)/sigma_x
    EI = (mu_x - f_xplus - xi)*scipy.stats.norm.cdf(Z) + sigma_x*scipy.stats.normal.pdf(Z)
    return EI