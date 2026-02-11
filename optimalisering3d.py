import numpy as np
import scipy.stats
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import copy
import pickle

#tar ikke høyde for symmetrien i at opt(x1,x2,x3) = opt(x2,x3,x1) osv
def loop(x_init, sim_func, acq_func, domain, tol=1e-3, debug=False, save_interval=1): 
    X_samples = np.atleast_2d(x_init)
    Y_samples = np.array([sim_func(x) for x in x_init])

    history = {'gpr':[],
               'acq_values': [],
               'x_next': [],
               'save_interval': save_interval}

    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True) #usikker på valg av kjerne

    acq_max = np.inf
    iteration = 0 #er nok feil i indekseringen
    while acq_max > tol:
        iteration += 1
        gpr.fit(X_samples, Y_samples)
        if iteration % save_interval == 0:
            history['gpr'].append(copy.deepcopy(gpr))
        mu, sigma = gpr.predict(domain, return_std=True)

        current_acq = acq_func(mu, sigma, np.max(Y_samples))
        if iteration % save_interval == 0:
            history['acq_values'].append(current_acq)

        idx_next = np.argmax(current_acq) #where acquisition function is largest
        x_next = domain[idx_next] 
        acq_max = current_acq[idx_next]
        score_next = sim_func(x_next)
        if iteration % save_interval == 0:
            history['x_next'].append(x_next)

        X_samples = np.vstack((X_samples, x_next.reshape(1,-1)))
        Y_samples = np.append(Y_samples, score_next)
        
        if debug:
            print(f'Next point: {x_next}, Function value:{round(score_next,3)}, Acquisition max: {acq_max}')
    #should apply symmetry of fire alarms: (x1,x2,x3) == (x2,x1,x3) osv.
    return X_samples, Y_samples, history

def expected_improvement(mu_x, sigma_x, f_xplus, xi=1e-3): #add another acquisition function?
    Z = (mu_x - f_xplus - xi)/sigma_x
    EI = (mu_x - f_xplus - xi)*scipy.stats.norm.cdf(Z) + sigma_x*scipy.stats.norm.pdf(Z)
    return EI

def sim_test3d(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    term1 = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    term2 = 100 * (x3 - x2**2)**2 + (1 - x2)**2
    
    return -(term1 + term2)

X, Y, Z = np.meshgrid(np.arange(0,2,1e-2),np.arange(0,2,1e-2),np.arange(0,2,1e-2))
domain3d = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

res_x3d, res_y3d, res_history3d = loop(
    [[0.5, 0.5, 0.5]], 
    sim_test3d, 
    expected_improvement, 
    domain3d, 
    tol=.3, 
    debug=True, 
    save_interval=1)

data_to_save = {
    'x': res_x3d,
    'y': res_y3d, #response
    'history': res_history3d,
    'domain': domain3d
}

with open('bo_results3d.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Results saved to bo_results3d.pkl")