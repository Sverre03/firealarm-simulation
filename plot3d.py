import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

filename = 'bo_results3d.pkl'

if os.path.exists(filename):
    print("Loading existing results...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    x = data['x']
    y = data['y']
    history = data['history']
    domain = data['domain']
    save_interval = history['save_interval']
    print('Data loaded.')
else:
    print('No saved data found. Run the simulation first.')


def plot_3d_levels(gpr, domain3d, res_x, n_slices=4):
    x_vals = np.unique(domain3d[:, 0])
    y_vals = np.unique(domain3d[:, 1])
    z_vals = np.unique(domain3d[:, 2])
    
    # 2. Select which Z-levels to show
    # We'll pick n_slices evenly spaced out along the Z axis
    idx_slices = np.linspace(0, len(z_vals) - 1, n_slices).astype(int)
    z_levels = z_vals[idx_slices]

    fig, axes = plt.subplots(1, n_slices, figsize=(20, 5), sharey=True)
    
    # Global scale for consistent colors
    mu_all = gpr.predict(domain3d)
    vmin, vmax = mu_all.min(), mu_all.max()

    for i, z_lev in enumerate(z_levels):
        ax = axes[i]
        
        mask = np.isclose(domain3d[:, 2], z_lev, atol=1e-5)
        slice_coords = domain3d[mask]
        
        mu_slice = gpr.predict(slice_coords)
        
        X_slice = slice_coords[:, 0].reshape(len(y_vals), len(x_vals))
        Y_slice = slice_coords[:, 1].reshape(len(y_vals), len(x_vals))
        Z_slice = mu_slice.reshape(len(y_vals), len(x_vals))
        
        cntr = ax.contourf(X_slice, Y_slice, Z_slice, levels=30, 
                           cmap='viridis', vmin=vmin, vmax=vmax)
        
        z_tol = (z_vals[1] - z_vals[0]) * 2
        nearby_samples = res_x[np.abs(res_x[:, 2] - z_lev) < z_tol]
        if len(nearby_samples) > 0:
            ax.scatter(nearby_samples[:, 0], nearby_samples[:, 1], 
                       c='red', edgecolors='white', s=40, label='Samples')

        ax.set_title(f"Slice Z = {z_lev:.2f}")
        ax.set_xlabel("X1")
        if i == 0: ax.set_ylabel("X2")

    # Single colorbar for the whole figure
    fig.colorbar(cntr, ax=axes.ravel().tolist(), label='GPR Prediction')
    plt.suptitle("3D Model Slices (Level Plots)", fontsize=16)
    plt.show()


def plot_4d_scatter_fast(gpr, domain3d, samples_x, resolution=30):
    x_vals = np.linspace(domain3d[:,0].min(), domain3d[:,0].max(), resolution)
    y_vals = np.linspace(domain3d[:,1].min(), domain3d[:,1].max(), resolution)
    z_vals = np.linspace(domain3d[:,2].min(), domain3d[:,2].max(), resolution)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    grid_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    W = gpr.predict(grid_coords)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    threshold = np.percentile(W, 90) 
    mask = W > threshold

    img = ax.scatter(X.ravel()[mask], Y.ravel()[mask], Z.ravel()[mask], 
                     c=W[mask], cmap='inferno', 
                     marker='o', s=10, alpha=0.3, depthshade=False)

    ax.scatter(samples_x[:, 0], samples_x[:, 1], samples_x[:, 2], 
               c='cyan', s=80, edgecolors='black', label='Samples', depthshade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.colorbar(img, ax=ax, label='Value')
    ax.set_title(f"3D Performance View ({len(X.ravel()[mask])} points shown)")
    plt.legend()
    plt.show()

last_gpr = history['gpr'][-1]
plot_3d_levels(last_gpr, domain, x, n_slices=3)
plot_4d_scatter_fast(last_gpr, domain, x, resolution=25)