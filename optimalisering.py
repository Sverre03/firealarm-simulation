import numpy as np
import scipy.stats
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import copy

def loop(x_init, sim_func, acq_func, domain, tol=1e-3, debug=False):
    X_samples = np.array(x_init).reshape(-1,1)
    Y_samples = np.array([sim_func(x) for x in x_init])

    history = {'gpr':[],
               'acq_values': [],
               'x_next': []}

    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5))

    acq_max = np.inf
    while acq_max > tol:
        gpr.fit(X_samples, Y_samples)
        history['gpr'].append(copy.deepcopy(gpr))
        mu, sigma = gpr.predict(domain.reshape(-1,1), return_std=True)

        current_acq = acq_func(mu, sigma, np.max(Y_samples))
        history['acq_values'].append(current_acq)

        idx_next = np.argmax(current_acq)
        x_next = domain[idx_next]
        acq_max = current_acq[idx_next]
        score_next = sim_func(x_next)
        history['x_next'].append(x_next)

        X_samples = np.vstack((X_samples, [[x_next]]))
        Y_samples = np.append(Y_samples, score_next)
        
        if debug:
            print(f'x_next: {x_next}\nscore_next:{score_next}')
    #nshould apply symmetry of fire alarms: (x1,x2,x3) == (x2,x1,x3) osv.
    return X_samples, Y_samples, history

def expected_improvement(mu_x, sigma_x, f_xplus, xi=0):
    Z = (mu_x - f_xplus - xi)/sigma_x
    EI = (mu_x - f_xplus - xi)*scipy.stats.norm.cdf(Z) + sigma_x*scipy.stats.norm.pdf(Z)
    return EI

def sim_test(x):
    return (x-1)**3+1-x

domain = np.arange(0,3,1e-3)
res_x, res_y, res_history = loop(np.array([0.5]), sim_test, expected_improvement, domain, tol=1e-3)
print(f'Maximum at ({res_x[np.argmax(res_y)]}, {max(res_y)}')

plt.scatter(res_x, res_y)
plt.plot(domain, sim_test(domain))
plt.show()