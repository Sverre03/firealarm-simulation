import numpy as np
import scipy.stats
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import copy

#tar ikke høyde for symmetrien i at opt(x1,x2,x3) = opt(x2,x3,x1) osv
def loop(x_init, sim_func,  acq_func, domain, tol=1e-3,  debug=False, save_interval=1, max_iterations=1000): 
    log_sim = lambda x: np.maximum(np.log(sim_func(x)), 1e-8)
    X_samples = np.atleast_2d(np.asarray(x_init, dtype=float))
    Y_samples = np.array([log_sim(x) for x in X_samples], dtype=float)

    if save_interval is None:
        should_save = lambda i: False
    else:
        should_save = lambda i: (i % save_interval == 0)
    
    history = {
        "gpr": [],
        "acq_values": [],
        "x_next": [],
        "save_interval": save_interval,
    }
    log_sim = lambda x: np.log(np.maximum(sim_func(x), 1e-8))
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)

    acq_max = np.inf
    iteration = 0

    while acq_max > tol and iteration < max_iterations:
        iteration += 1
        
        should_save_now = should_save(iteration)

        gpr.fit(X_samples, Y_samples)
        if should_save_now:
            history["gpr"].append(copy.deepcopy(gpr))

        mu, sigma = gpr.predict(domain, return_std=True)
        current_acq = acq_func(mu, sigma, float(np.max(Y_samples)))
        current_acq = np.where(np.isnan(current_acq), -np.inf, current_acq)

        if should_save_now:
            history["acq_values"].append(current_acq)

        idx_next = int(np.argmax(current_acq))
        x_next = domain[idx_next]
        acq_max = float(current_acq[idx_next])

        score_next = float(log_sim(x_next))

        if should_save_now:
            history["x_next"].append(x_next)

        X_samples = np.vstack((X_samples, x_next.reshape(1,-1)))
        Y_samples = np.append(Y_samples, score_next)

        if debug:
            print(f"iter={iteration} acq_max={acq_max:.6g} log-coverage={score_next:.2f}")

    return X_samples, Y_samples, history


def expected_improvement(mu_x, sigma_x, f_xplus, xi=1e-3):
    """Expected Improvement acquisition function (safe for sigma=0)."""
    sigma_safe = np.maximum(np.asarray(sigma_x, dtype=float), 1e-12)
    mu_x = np.asarray(mu_x, dtype=float)

    improvement = mu_x - float(f_xplus) - float(xi)
    z = improvement / sigma_safe

    ei = improvement * scipy.stats.norm.cdf(z) + sigma_safe * scipy.stats.norm.pdf(z)
    ei[sigma_x <= 1e-12] = 0.0
    return ei
