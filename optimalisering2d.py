import numpy as np
import scipy.stats
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import copy
import pickle

def loop(x_init, sim_func, acq_func, domain, tol=1e-3, debug=False, save_interval=10):
    X_samples = np.atleast_2d(x_init)
    Y_samples = np.array([sim_func(x) for x in x_init])

    history = {'gpr':[],
               'acq_values': [],
               'x_next': [],
               'save_interval': save_interval}

    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True) #usikker på valg av kjerne

    acq_max = np.inf
    iteration = 0
    while acq_max > tol:
        iteration += 1
        gpr.fit(X_samples, Y_samples)
        if iteration % save_interval == 0:
            history['gpr'].append(copy.deepcopy(gpr))
        mu, sigma = gpr.predict(domain, return_std=True)

        current_acq = acq_func(mu, sigma, np.max(Y_samples))
        if iteration % save_interval == 0:
            history['acq_values'].append(current_acq)

        idx_next = np.argmax(current_acq)
        x_next = domain[idx_next]
        acq_max = current_acq[idx_next]
        score_next = sim_func(x_next)
        if iteration % save_interval == 0:
            history['x_next'].append(x_next)

        X_samples = np.vstack((X_samples, x_next.reshape(1,-1)))
        Y_samples = np.append(Y_samples, score_next)
        
        if debug:
            print(f'x_next: {x_next}\nscore_next:{score_next}')
    #should apply symmetry of fire alarms: (x1,x2,x3) == (x2,x1,x3) osv.
    return X_samples, Y_samples, history

def expected_improvement(mu_x, sigma_x, f_xplus, xi=1e-3):
    Z = (mu_x - f_xplus - xi)/sigma_x
    EI = (mu_x - f_xplus - xi)*scipy.stats.norm.cdf(Z) + sigma_x*scipy.stats.norm.pdf(Z)
    return EI

def sim_test2d(x):
    return (x[0]-1)**3+1-x[0] + (x[1]-1)**2

def six_hump_camel(x):
    x1, x2 = x[0], x[1]
    val = (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
    return -val

X, Y = np.meshgrid(np.arange(-3,3,1e-2),np.arange(-2,2,1e-2))
domain2d = np.vstack([X.ravel(), Y.ravel()]).T

res_x2d, res_y2d, res_history2d = loop([[0.5, 0.5]], six_hump_camel, expected_improvement, domain2d, tol=.3, debug=True)

data_to_save = {
    'x': res_x2d,
    'y': res_y2d,
    'history': res_history2d,
    'domain': domain2d
}

with open('bo_results.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Results saved to bo_results.pkl")