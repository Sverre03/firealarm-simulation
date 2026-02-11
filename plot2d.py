import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = 'bo_results.pkl'

if os.path.exists(filename):
    print("Loading existing results...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    res_x2d = data['x']
    res_y2d = data['y']
    res_history2d = data['history']
    domain2d = data['domain']
    save_interval = res_history2d['save_interval']
else:
    print("No saved data found. Run the simulation first.")

last_gpr = res_history2d['gpr'][-1]


side = int(np.sqrt(len(domain2d))) 
xcount = 600
ycount = 400
X_grid = domain2d[:,0].reshape(ycount, xcount)
Y_grid = domain2d[:,1].reshape(ycount, xcount)

mu, sigma = last_gpr.predict(domain2d, return_std=True)
Z = mu.reshape(ycount, xcount)

plt.contourf(X_grid, Y_grid, Z)
plt.scatter(res_x2d[:,0], res_x2d[:,1], c='red', label='Samples')
plt.show()