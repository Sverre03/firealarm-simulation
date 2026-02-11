import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = 'bo_results.pkl'

if os.path.exists(filename):
    print("Loading existing results...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    x = data['x']
    y = data['y']
    history = data['history']
    domain2d = data['domain']
    save_interval = history['save_interval']
else:
    print("No saved data found. Run the simulation first.")

last_gpr = history['gpr'][-1]


xcount = len(np.unique(domain2d[:, 0]))
ycount = len(np.unique(domain2d[:, 1]))

X_grid = domain2d[:,0].reshape(ycount, xcount)
Y_grid = domain2d[:,1].reshape(ycount, xcount)

mu, sigma = last_gpr.predict(domain2d, return_std=True)
Z = mu.reshape(ycount, xcount)

plt.contourf(X_grid, Y_grid, Z,)
plt.scatter(x[:,0], x[:,1], c='red', label='Samples')
plt.show()

fig, ax = plt.subplots(figsize=(10, 7))

all_acq = np.concatenate(history['acq_values'])

first_acq = history['acq_values'][0].reshape(ycount, xcount)


scat_past = ax.scatter([], [], c='white', edgecolors='black', s=30, label='Past Samples')
scat_next = ax.scatter([], [], c='cyan', marker='*', s=200, label='Next Choice')

global_min = np.min(all_acq)
global_max = np.max(all_acq)
cont = ax.contourf(X_grid, Y_grid, first_acq, levels=50, cmap='magma', vmin=global_min, vmax=global_max)

norm = plt.Normalize(vmin=global_min, vmax=global_max)
sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Expected Improvement (EI)')
def update(frame):
    ax.clear() 
    
    acq_full = history['acq_values'][frame]
    acq_rect = acq_full.reshape(ycount, xcount)
    
    cont = ax.contourf(X_grid, Y_grid, acq_rect, levels=50, cmap='magma', vmin=global_min, vmax=global_max)
    
    current_samples = x[:frame+1]
    ax.scatter(current_samples[:, 0], current_samples[:, 1], 
               c='white', edgecolors='black', s=30, label='Past Samples')
    
    if frame < len(history['x_next']):
        next_pt = history['x_next'][frame]
        ax.scatter(next_pt[0], next_pt[1], c='cyan', s=60, 
                   edgecolors='black', label='Point Chosen (Next)')

    interval_val = history.get('save_interval', 1) # Fallback to 1 if not found
    ax.set_title(f"Acquisition Function - Iteration {frame * interval_val}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc='upper right')
    
    return cont

ani = FuncAnimation(fig, update, frames=len(history['acq_values']), 
                    interval=1000, repeat=True, blit=False)

plt.tight_layout()
plt.show()