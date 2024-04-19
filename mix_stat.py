# -*- coding: utf-8 -*-
"""
A small toolkit for playing with the statistics of a solid mixing process
For further information see e.g. Stieß2009, Mechanische Verfahrenstechnik (in german)
@author: Frank Rhein, frank.rhein@kit.edu
"""
#%% IMPORTS
import numpy as np  # Matrix operations / general functionality
import matplotlib.pyplot as plt  # Plots and visualization
from scipy.stats import ncx2  # Non-central Chi2 distribution

#%% SUB-FUNCTIONS
def generate_grid(n_grid=100, case='random', c_A=0.5):
    if not (0 <= c_A <= 1):
        print('ERROR: Provided c_A is invalid. Needs to be in range [0,1]')
        return
    
    # M is an integer matrix with n_grid x n_grid points and values in the range of [0,n_comp)
    # Values are either 0 or 1 (0 = component A, 1 = component B)
    if case == 'random':
        M = np.random.choice([0, 1], (n_grid, n_grid), p=[c_A, 1 - c_A])
    elif case == 'ordered':
        M = np.ones(n_grid**2).astype(int)
        M[0:int(np.floor(c_A * n_grid**2))] = 0
        M = np.reshape(M, (n_grid, n_grid))
    
    return M

def visualize_M(M,
                ax=None,
                fig=None,
                close=False,
                title=None,
                Z=None,
                expname=None):

    if close:
        plt.close('all')
    
    # Create figure and axis if none are provided
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    # 2D plot of M
    ax.pcolor(M, cmap='summer')
    ax.set_axis_off()
    if title is not None: ax.set_title(title)
    
    # Visualize sample size if N is provided
    if Z is not None:
        _, idx = pick_sample(M, Z=Z, coords=None)
        for i in idx:
            ax.scatter(i[0], i[1], marker='o', s=10, color='k', label='sample size')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:1], labels[:1])
      
    plt.tight_layout()
    
    if expname is not None:
        #plt.savefig('export/'+expname+'.png',dpi=300)
        plt.savefig('export/' + expname + '.pdf')
    
    return ax, fig

def visualize_s2_t(t,
                   s2,
                   ax=None,
                   fig=None,
                   close=False,
                   title=None,
                   col='midnightblue',
                   lbl=None,
                   y_lines=[],
                   y_line_labels=[],
                   data='s2'):

    if close:
        plt.close('all')
    
    # Create figure and axis if none are provided
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    if data == 's2':
        ax.scatter(t, s2, marker='o', color=col, zorder=2, label=lbl)
        ax.plot(t, s2, color=col, zorder=2)
    elif data == 'b2':
        ax.fill_between(t, s2, alpha=0.3, color=col, label=lbl)
    
    for i, y in enumerate(y_lines):
        ax.axhline(y, color='k')
        ax.text(ax.get_xlim()[1],
                y,
                y_line_labels[i],
                horizontalalignment='left',
                verticalalignment='center')
    
    ax.set_xlabel('Mixing time $t$ / $a.u.$')
    ax.set_ylabel('Variance $s^2$ / $-$')
    # ax.set_ylim([1e-3,1])
    # ax.set_yscale('log')
    ax.grid(True)
    if lbl is not None: ax.legend(loc='upper right', framealpha=1)
    
    plt.tight_layout()
    
    return ax, fig

def swap_n_elements_dz(M, n, deadzone=0.1):
    # Consistency Check
    if not 0 <= deadzone < 1:
        print('ERROR: Deadzone value needs to be between 0 and 1. Exiting..')
        return
    
    for i in range(n):
        coords_1 = tuple(np.random.randint(0, int((1-deadzone)*len(M[:, 0])), 2))
        coords_2 = tuple(np.random.randint(0, int((1-deadzone)*len(M[:, 0])), 2))
        M[coords_1], M[coords_2] = M[coords_2], M[coords_1]
    
    return M

def swap_n_elements(M, n, deadzone=None):

    for i in range(n):
        coords_1 = tuple(np.random.randint(0, len(M[:, 0]), 2))
        coords_2 = tuple(np.random.randint(0, len(M[:, 0]), 2))
        M[coords_1], M[coords_2] = M[coords_2], M[coords_1]
    
    return M

def demix_n_elements(M, n):
    
    for i in range(n):
        # Choose a random index in the vertical mid-third of M
        y = np.random.randint(int(len(M[:, 0])/3), int(2*len(M[:, 0])/3))
        x = np.random.randint(0,len(M[:, 0]))
        # Chosen value is 0 --> swap downwards
        if M[y,x] == 0:
            y2 = np.random.randint(0, int(len(M[:, 0])/3))
            x2 = np.random.randint(0,len(M[:, 0]))
            M[y,x], M[y2,x2] = M[y2,x2], M[y,x]
        
        # Else --> swap upwards
        else:
            y2 = np.random.randint(int(2*len(M[:, 0])/3),len(M[:, 0]))
            x2 = np.random.randint(0,len(M[:, 0]))
            M[y,x], M[y2,x2] = M[y2,x2], M[y,x]
    
    return M
    
def return_chi2(f, S):
    nc = 0.01
    x = np.linspace(ncx2.ppf(0.01, f, nc), ncx2.ppf(0.99, f, nc), 100)
    return x[np.argmin((ncx2.cdf(x, f, nc) - S)**2)]


def pick_sample(M, Z=1, coords=None):

    m = len(M[:, 0]) # Extract size of mixture matrix
    
    if Z is not int:
        Z = int(Z)
      
    if coords is None:
        coords = np.random.randint(0, m, 2)
      
    sample = []  # Empty list for samples
    idx = []  # Empty list for indices

    indices = np.argwhere(np.ones((m, m)))
    distances = np.sqrt(np.sum((indices - coords) ** 2, axis=1))
    sorted_indices = indices[np.argsort(distances)]
    for i in sorted_indices:
        if (i[0] >= 0 and i[0] < m) and (i[1] >= 0 and i[1] < m):
            idx.append(i)
            sample.append(M[i[0],i[1]])
        if len(idx) == Z:
            break

    return sample, idx

def s2_from_samples(samples, conf=95, P=None):
    # Samples is a 3D array with
    # Dimension 0: timestep
    # Dimension 1: sample no. of given timestep
    # Dimension 2: actual sample entries
      
    # Initialize s2 and b2 array (1D: time)
    s2 = np.zeros(samples.shape[0])
    b2 = np.zeros(samples.shape[0])
      
    S = 1 - conf / 100  # Value S for confidence interval
      
    # Transform 3D samples array into 2D X array (number concentraion of A)
    X = np.zeros((samples.shape[0], samples.shape[1]))
      
    # Loop through sampling times
    for t in range(samples.shape[0]):
        # Loops through all samples
        for n in range(samples.shape[1]):
            # Count number of A in sample (entry is 0)
            N_A_s = len(samples[t, n, :]) - np.count_nonzero(samples[t, n, :])
            # Calculate concentration of A in sample
            X[t, n] = N_A_s / len(samples[t, n, :])
        # Calculate variance. If true value is not given --> Use s_(n-1)
        if P is None:
            s2[t] = np.sum((X[t, :] - np.mean(X[t, :]))**2) / (len(X[t, :]) - 1)
            chi2 = return_chi2(len(X[t, :]) - 1, S)
            b2[t] = (len(X[t, :]) - 1) * s2[t] / chi2
        else:
            s2[t] = np.sum((X[t, :] - P)**2) / len(X[t, :])
            chi2 = return_chi2(len(X[t, :]), S)
            b2[t] = len(X[t, :]) * s2[t] / chi2

    return s2, b2


def perform_experiment(t_exp,
                       c_A,
                       t_samples=[],
                       num_samples_per=3,
                       n_grid=100,
                       Z=5,
                       D=1e-4,
                       visualize_mix=True,
                       deadzone=None):

    if t_exp is not int:
      t_exp = int(t_exp)
    
    # Consistency Check (n_grid > N)
    if not n_grid**2 >= Z:
        print('ERROR: Sample size is larger than mixture. Exiting..')
        return
        
    # Initialize samples array.
    # Dimension 0: timestep
    # Dimension 1: sample no. of given timestep
    # Dimension 2: actual sample entries
    samples = np.ones((len(t_samples), num_samples_per, Z))
    
    # Generate mixture matrix M
    M = generate_grid(n_grid=n_grid, case='ordered', c_A=c_A)
    
    # Visualize initial state of M
    if visualize_mix:
        ax, fig = visualize_M(M,
                              close=True,
                              title='Initial State of Mixture ($t=0$)',
                              expname='M_before')
    
    # Calculate number of swaps per timestep (depending on gridsize). Minimum = 1!
    num_swaps = max(1, int((n_grid**2) * D))
    
    cnt_t = 0  # Initialize sampling time counter
    
    # Loop through all timesteps
    for i in range(t_exp):
        # Swap corresponding number of times
        if deadzone is None:
            M = swap_n_elements(M, num_swaps)  
        else:
            M = swap_n_elements_dz(M, num_swaps, deadzone)
            
        # Time for some samples!
        if i in t_samples:
            for n in range(num_samples_per):
                tmp_sample, _ = pick_sample(M, Z)
                samples[cnt_t, n, :] = tmp_sample
            
            cnt_t += 1  # increase counter
    
    # Visualize final state of M
    if visualize_mix:
      ax, fig = visualize_M(M,
                            title=f'Final State of Mixture ($t={t_exp:.1e}$)',
                            Z=Z,
                            expname='M_after')
    
    return samples, M

def perform_demixing(M,
                     t_exp,
                     t_samples=[],
                     num_samples_per=3,
                     Z=5,
                     D_demix=1e-4,
                     visualize_mix=True):

    if t_exp is not int:
      t_exp = int(t_exp)
    
    # Consistency Check (n_grid > N)
    if not n_grid**2 >= Z:
        print('ERROR: Sample size is larger than mixture. Exiting..')
        return
        
    # Initialize samples array.
    # Dimension 0: timestep
    # Dimension 1: sample no. of given timestep
    # Dimension 2: actual sample entries
    samples = np.ones((len(t_samples), num_samples_per, Z))
    
    # Visualize initial state of M
    if visualize_mix:
        ax, fig = visualize_M(M,
                              close=False,
                              title='State of Mixture before Demixing ($t=0$)',
                              expname='M_before_demix')
    
    # Calculate number of demixing swaps (0 is allowed)
    num_demix = int((n_grid**2) * D)
    
    cnt_t = 0  # Initialize sampling time counter
    
    # Loop through all timesteps
    for i in range(t_exp):
        
        # Demixing swaps if D_demix is not 0
        if D_demix != 0:
            M = demix_n_elements(M, num_demix)
            
        # Time for some samples!
        if i in t_samples:
            for n in range(num_samples_per):
                tmp_sample, _ = pick_sample(M, Z)
                samples[cnt_t, n, :] = tmp_sample
            
            cnt_t += 1  # increase counter
    
    # Visualize final state of M
    if visualize_mix:
      ax, fig = visualize_M(M,
                            title=f'State of Mixture after Demixing ($t={t_exp:.1e}$)',
                            Z=Z,
                            expname='M_after_demix')
    
    return samples, M


#%% MAIN
if __name__ == '__main__':

    # General Parameters
    t_exp = 5000  # in timesteps
    num_t_samples = 15  # numer of linearly spaces sampling times
    num_samples_per = 20  # numer of samples per timestep
    n_grid = 50  # grid size (n_grid^2 particles)
    Z = 50  # sampling size
    c_A = 0.5  # relative concentration component A
    D = 5e-4  # "Diffusion coefficient" (number of random swaps per particle and timestep)
    alpha = 75  # Confidence limit for variance (in %)
    
    # Additional (optional) settings
    analyze_samples = True # Do you want to analyze samples after experiment?
    
    # Dead-Zone and Demixing tests
    deadzone = None # Define a deadzone in the mixer (range [0,1] as % of mixer length). None to turn off
    D_demix = 1e-5 # "Diffusion coefficient" for demixing. 0 to turn demixing off 
    t_demix = 2000 # in timesteps
    
    t_samples = list(np.linspace(0, t_exp - 1, num_t_samples).astype(int))  # Generate list of sampling times
    # Hint for later: np.logspace?
    
    # Run the mixing experiment!
    samples, M = perform_experiment(t_exp=t_exp,
                                    c_A=c_A,
                                    t_samples=t_samples,
                                    num_samples_per=num_samples_per,
                                    n_grid=n_grid,
                                    Z=Z,
                                    D=D,
                                    visualize_mix=True,
                                    deadzone=deadzone)
    
    # Simulate demixing after experiment
    if D_demix > 0:
        # Generate list of sampling times
        t_samples_demix = list(np.linspace(0, t_demix - 1, num_t_samples).astype(int))
        
        samples_dm, M = perform_demixing(M,
                                         t_exp=t_demix,
                                         t_samples=t_samples_demix,
                                         num_samples_per=num_samples_per,
                                         Z=Z,
                                         D_demix=D_demix,
                                         visualize_mix=True)
        # Append samples to samples-array and corresponding times (add t_exp first)
        samples = np.append(samples, samples_dm, axis=0)
        t_samples += [x+t_exp for x in t_samples_demix]
        
    if analyze_samples:
        # Calculating s2 and b2 from samples (P is given)
        s2, b2 = s2_from_samples(samples, conf=alpha, P=c_A)
        # Calculating s2_n1 and b2_n1 from samples (P is NOT given)
        s2_n1, b2_n1 = s2_from_samples(samples, conf=alpha, P=None)
           
        # Calculating sigma_0 and sigma_Z
        sig_0 = c_A * (1 - c_A)
        sig_z = sig_0 * (1 / Z)
           
        # Visualizing s2 with known true concentration P
        ax, fig = visualize_s2_t(t_samples,
                                 s2,
                                 col='midnightblue',
                                 lbl='$s^2(t)$',
                                 y_lines=[sig_0, sig_z],
                                 y_line_labels=['$\sigma_0^2$', '$\sigma_z^2$'])
        ax, fig = visualize_s2_t(t_samples,
                                 b2,
                                 col='firebrick',
                                 ax=ax,
                                 fig=fig,
                                 lbl='$b^2(t)$',
                                 data='b2')
           
        fig.savefig(f'export/s2_t.pdf')
           
        # Visualizing s2_n-1 without known true concentration P
        ax, fig = visualize_s2_t(t_samples,
                                 s2_n1,
                                 col='midnightblue',
                                 lbl='$s_{n-1}^2(t)$',
                                 y_lines=[sig_0, sig_z],
                                 y_line_labels=['$\sigma_0^2$', '$\sigma_z^2$'])
        ax, fig = visualize_s2_t(t_samples,
                                 b2_n1,
                                 col='firebrick',
                                 ax=ax,
                                 fig=fig,
                                 lbl='$b_{n-1}^2(t)$',
                                 data='b2')
           
        fig.savefig(f'export/s2_n1_t.pdf')
  
