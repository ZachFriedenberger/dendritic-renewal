import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import matplotlib as mpl
import pandas as pd


plt.rcParams.update({'font.size': 8})

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}

mpl.rcParams.update(new_rc_params)

from dendritic_renewal_model import input_potential, params_reset, input_potential_gauss

def save_plot_data_csv(data_frame, filename):
    
    data_frame.to_csv('../data/' + filename, index=False)
    
    return


def plot_input_potential(input_potential, filename, xd_I=None):
    """
    Create a plot of the input potential as a function of x.
    
    Returns
    -------
       A plot saved as a jpeg and svg file. 

    """
    params = params_reset()
    params['I_s'] = 5
    params['I_d'] = 5
    
    if xd_I: 
        params['xd_I'] = xd_I
    
    
    I_s = params['I_s']
    I_d = params['I_d']
    x_d = params['x_d']
    
    
    L = params['L']
    
    X = np.linspace(0, L, 101)
    h = input_potential(X, params)
    
    plt.figure(figsize=(1, 0.5))
    plt.plot(X, h, 'k')
    sns.despine()
    plt.ylabel('$h(x)$')
    plt.xlabel('$x$')
    plt.yticks([0, 2.5, 5])
    plt.xticks([0, 5, 10])
    plt.ylim([0, 5])
    plt.xlim([-.3, L])
    plt.vlines(x_d, 0, I_d/2, linestyle='--', color='tab:red', label='$I_{d}$')
    plt.vlines(0, 0, I_s, linestyle='--', color='tab:blue', label='$I_{s}$')
    plt.title('Subthreshold')
    
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('../results/' + filename + '.svg', dpi=300)
    
    data_frame = pd.DataFrame({"x" : X, "h" : h})
    save_plot_data_csv(data_frame, filename=filename + '.csv')
    
    return

def plot_gaussian_input(xd_I):
    
    params = params_reset()
    
    #xd = params['xd_I']
    sigma = params['sigma_I']
    L = params['L']
    I_d = 5
    
    x = np.linspace(0, L, 100)

    y = I_d*np.exp(-(x-xd_I)**2 /(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    
    plt.figure(figsize=(1, 0.5))
    plt.plot(x, y,'--', color='k')
    sns.despine()
    plt.ylabel('$I_{d}(x)$')
    plt.xlabel('$x$')
    plt.yticks([0, 2.5, 5])
    plt.xticks([0, 5, 10])
    plt.ylim([0, 5])
    plt.xlim([-.3, L])
    plt.title('Input')
    

    plt.tight_layout()
    filename = f'gaussian_input_xd_{xd_I}'
    plt.savefig('../results/' + filename + '.svg')
    
    return

def plot_gaussian_filter():
        
    params = params_reset()
    
    mu = params['mu']
    sigma = params['sigma']
    L = params['L']
    
    x = np.linspace(0, L, 100)

    y = np.exp(-(x-mu)**2 /(2*sigma**2))
    
    plt.figure(figsize=(1, 0.5))
    plt.plot(x, y,'--', color='tab:orange')
    sns.despine()
    plt.ylabel('$F(x)$')
    plt.xlabel('$x$')
    plt.yticks([0, 0.5, 1])
    plt.xticks([0, 5, 10])
    plt.ylim([0, 1])
    plt.xlim([-.3, L])
    plt.title('Input')
    

    plt.tight_layout()
    filename = 'gaussian_filter'
    plt.savefig('../results/' + filename + '.svg')
    
    
    return

def plot_effective_input_potential(input_potential, filename):
    """
    Create a plot of the effective input potential as a function of x. The effective potential is the input potential scaled by an exponential function.
    
    Returns
    -------
       A plot saved as a jpeg and svg file. 
       
    """    
    params = params_reset()
    params['I_s'] = 5
    params['I_d'] = 5
    
    I_s = params['I_s']
    I_d = params['I_d']
    x_d = params['x_d']
    L = params['L']
    gamma = params['gamma']
    
    X = np.linspace(0, L, 101)
    h = input_potential(X, params)
    
    
    scale = np.exp(gamma*(X-L))
    
    
    plt.figure(figsize=(3, 1.5))
    plt.plot(X, scale*h, 'k')
    sns.despine()
    plt.ylabel(r'$\tilde{h}(x)$')
    plt.xlabel('$x$')
    plt.yticks([0, 0.5, 1])
    plt.ylim([0, 1])
    
    plt.xlim([-.3, L])
    
    scale_xd = scale[np.where(X==x_d)]
    plt.vlines(x_d, 0, scale_xd*I_d/2, linestyle='--', color='tab:red', label='$I_{d}$')
    plt.vlines(0, 0, scale[0]*I_s, linestyle='--', color='tab:blue', label='$I_{s}$')
    plt.title('Subthreshold')
    
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('../results/' + filename + '.svg')
    
    return

def plot_effective_gauss_input_potential(input_potential, filename, xd_I):
    """
    Create a plot of the effective input potential as a function of x. The effective potential is the input potential scaled by an exponential function.
    
    Returns
    -------
       A plot saved as a jpeg and svg file. 
       
    """    
    params = params_reset()
    params['I_s'] = 5
    params['I_d'] = 5
    params['xd_I'] = xd_I
    L = params['L']
    mu = params['mu']
    sigma = params['sigma']

    X = np.linspace(0, L, 101)
    h = input_potential(X, params)
    
    scale = np.exp(-(X-mu)**2 /(2*sigma**2))
    
    plt.figure(figsize=(1, 0.5))
    plt.plot(X, scale*h, 'k')
    sns.despine()
    plt.ylabel(r'$\tilde{h}(x)$')
    plt.xlabel('$x$')
    plt.yticks([0, 2, 4])
    plt.ylim([0, 4])
    
    plt.xlim([-.3, L])
    plt.title('Subthreshold')
    
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('../results/' + filename + '.svg')
    
    return


def plot_A_inf(A_inf_tmp, A_inf_passive, I_s_vec, I_d_vec, title, filename):
    

    color='tab:blue'
    alpha = [1, 0.7, 0.5]
    
    plt.figure(figsize=(1, 0.75))
    plt.title(title)
    for j, I_d in enumerate(I_d_vec):
        plt.plot(I_s_vec, A_inf_tmp[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    sns.despine()
    plt.xticks([0, 2.5, 5])
    #plt.yticks([0, 50, 100, 150])
    #plt.ylim([0, 160])
    plt.xlabel(r'$I_{s}$')
    plt.ylabel(r'$A_{\infty}$ [Hz]')
        
    plt.plot(I_s_vec, A_inf_passive[0,:], color='k', linestyle='--', label='Passive')
    
    plt.legend(frameon=False)
    plt.tight_layout()
    
    plt.savefig(filename)
    
    return


def plot_A_inf_versus_input_locaiton(A_inf_tmp, A_inf_passive, xd_vec, I_d_vec, title, filename):
    

    color='tab:blue'
    alpha = [1, 0.7, 0.5]
    
    plt.figure(figsize=(1, 0.75))
    plt.title(title)
    for j, I_d in enumerate(I_d_vec):
        plt.plot(xd_vec, A_inf_tmp[j, :]-A_inf_passive[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    sns.despine()
    plt.xticks([5, 10])
    plt.yticks([0, 15, 30])
    plt.ylim([0, 30])
    plt.xlabel(r'$x_{d}$')
    plt.ylabel(r'$\Delta A_{\infty}$ [Hz]')
        
    #plt.plot(xd_vec, A_inf_passive[0,:], color='k', linestyle='--', label='Passive')
    
    plt.legend(frameon=False)
    plt.tight_layout()
    
    plt.savefig(filename)
    
    return


def plot_A_inf_monte_carlo_residual(A_inf_data, A_inf_mc_data, A_inf_mc_error, title, filename):
    """ Plot stationary firing rate vs. somatic input for varying dendritic inputs. Monte carlo points are plotted. 
    Residuals of points are in subplot above.

    Args:
        A_inf_data (TYPE): DESCRIPTION.
        A_inf_mc_data (TYPE): DESCRIPTION.
        A_inf_mc_error (TYPE): DESCRIPTION.
        title (TYPE): DESCRIPTION.
        filename (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    I_d_mc_vec = np.linspace(0, 5, 3)
    I_s_mc_vec = np.linspace(0, 5, 5)
    
    
    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(3, 3))
    
    for j, I_d in enumerate(I_d_vec):
        a1.plot(I_s_vec, A_inf_data[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
    
        
        
    indx = []
    for i, I_s_mc in enumerate(I_s_mc_vec):
        indx.append(np.where(I_s_vec==I_s_mc)[0][0])
        
    for j, I_d_mc in enumerate(I_d_mc_vec):
        a1.plot(I_s_mc_vec, A_inf_mc_data[j, :], '.', color='k')
        
        residual = A_inf_mc_data[j, :] - A_inf_data[j, indx]
        a0.plot(I_s_mc_vec, residual, '.', color='k')
        a0.errorbar(I_s_mc_vec, residual, A_inf_mc_error[j, :], fmt='None', color='tab:red')
    
    
    a0.hlines(0, 0, 5, linestyle='--', color='k', linewidth=1)    
    
    a1.set_xlabel(r'$I_{s}$')
    a1.set_ylabel(r'$A_{\infty}$ [Hz]')
    a1.set_xticks([0, 2.5, 5])
    a0.set_xticks([0, 2.5, 5])
    a0.set_yticks([-1, 0, 1])
    a1.set_yticks([0, 50, 100, 150])
    
    a0.set_xlim([-0.2, 5.2])
    a1.set_xlim([-0.2, 5.2])
    
    
    a0.set_ylabel('Residuals')
    
    a0.title.set_text(title)
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig(filename)
    
    return

def plot_A_inf_monte_carlo(A_inf_data, A_inf_mc_data, A_inf_mc_error, title, filename, A_inf_data_passive=None):
    """
    Plot stationary firing rate vs. somatic input for varying dendritic inputs. Monte carlo points are plotted on top.

    Args:
        A_inf_data (TYPE): DESCRIPTION.
        A_inf_mc_data (TYPE): DESCRIPTION.
        A_inf_mc_error (TYPE): DESCRIPTION.
        title (TYPE): DESCRIPTION.
        filename (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    I_d_mc_vec = np.linspace(0, 5, 3)
    I_s_mc_vec = np.linspace(0, 5, 5)
    
    
    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    plt.figure(figsize=(1.7, 1.5))
    
    for j, I_d in enumerate(I_d_vec):
        plt.plot(I_s_vec, A_inf_data[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    if A_inf_data_passive is not None: 
        plt.plot(I_s_vec, A_inf_data_passive[0, :], label='Passive', color='k', linestyle='--')
    
    for j, I_d_mc in enumerate(I_d_mc_vec):
        plt.plot(I_s_mc_vec, A_inf_mc_data[j, :], '.', color='k')
        
    
    plt.xlabel(r'$I_{s}$')
    plt.ylabel('Frequency [Hz]')
    plt.xticks([0, 2.5, 5])
    plt.yticks([0, 50, 100])
    plt.xlim([-0.2, 5.2])
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig('../results/' + filename + '.svg', dpi=300)
    
    # save source image data in csv
    dic = {"A_inf_data" : A_inf_data, "A_inf_mc_data" : A_inf_mc_data, "A_inf_data_passive":A_inf_data_passive}
    dic["A_inf_data"] = [tuple(scores) for scores in dic["A_inf_data"]]
    dic["A_inf_data_passive"] = [tuple(scores) for scores in dic["A_inf_data_passive"]]
    dic["A_inf_mc_data"] = [tuple(scores) for scores in dic["A_inf_mc_data"]]
    data_frame = pd.DataFrame.from_records(dic)
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_data.csv')
    
    data_frame = pd.DataFrame.from_records({"I_s_vec" : I_s_vec})
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_Isvec.csv')
    
    data_frame = pd.DataFrame.from_records({"I_d_vec" : I_d_vec})
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_Idvec.csv')
    
    data_frame = pd.DataFrame.from_records({"I_s_mc_vec" : I_s_mc_vec})
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_Ismcvec.csv')
    
    data_frame = pd.DataFrame.from_records({"I_d_mc_vec" : I_d_mc_vec})
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_Idmcvec.csv')
    
    
    return

def plot_kernel_changes(param1_vec, param1_key, param2_vec, param2_key):
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    for param1 in param1_vec:
        for param2 in param2_vec:
            
            params = params_reset()
            params[param1_key] = param1
            params[param2_key] = param2
            
            
            with open('../data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'rb') as f:
                A_inf_tmp = pickle.load(f)
            
            
            with open('../data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'rb') as f:
                A_inf_passive = pickle.load(f)
            
            title = param1_key + '=' + str(param1) + ', ' + param2_key + '=' + str(param2)
            filename = '../results/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.svg'
            plot_A_inf(A_inf_tmp, A_inf_passive, I_s_vec, I_d_vec, title, filename)
    
    
    return

def plot_kernel_changes_semilog(param1_vec, param1_key, param2_vec, param2_key):
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    for param1 in param1_vec:
        for param2 in param2_vec:
            
            params = params_reset()
            params[param1_key] = param1
            params[param2_key] = param2
            
            
            with open('../data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'rb') as f:
                A_inf_tmp = pickle.load(f)
            
            
            with open('../data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'rb') as f:
                A_inf_passive = pickle.load(f)
            
            title = param1_key + '=' + str(param1) + ', ' + param2_key + '=' + str(param2)
            filename = '../results/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_semilog.svg'
            plot_A_inf(np.log(A_inf_tmp), np.log(A_inf_passive), I_s_vec, I_d_vec, title, filename)
    
    
    return

def plot_kernel_changes_cv(param1_vec, param1_key, param2_vec, param2_key):
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    for param1 in param1_vec:
        for param2 in param2_vec:
            
            params = params_reset()
            params[param1_key] = param1
            params[param2_key] = param2
            
            
            with open('../data/cv_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'rb') as f:
                cv_tmp = pickle.load(f)
            
            
            with open('../data/cv' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'rb') as f:
                cv_passive = pickle.load(f)
            
            title = param1_key + '=' + str(param1) + ', ' + param2_key + '=' + str(param2)
            filename = '../results/cv_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.svg'
            plot_cv2(cv_tmp, cv_passive, I_s_vec, I_d_vec, title, filename)
    
    
    return

def plot_kernel_changes_cv_semilog(param1_vec, param1_key, param2_vec, param2_key):
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    for param1 in param1_vec:
        for param2 in param2_vec:
            
            params = params_reset()
            params[param1_key] = param1
            params[param2_key] = param2
            
            
            with open('../data/cv_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'rb') as f:
                cv_tmp = pickle.load(f)
            
            
            with open('../data/cv' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'rb') as f:
                cv_passive = pickle.load(f)
            
            title = param1_key + '=' + str(param1) + ', ' + param2_key + '=' + str(param2)
            filename = '../results/cv_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_semilog.svg'
            plot_cv2(np.log(cv_tmp), np.log(cv_passive), I_s_vec, I_d_vec, title, filename)
    
    
    return

def plot_spatial_dendrite(xd):
    
    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 21)
    

            
    with open('../data/integral_spatial_active_xd_' + str(xd) + '.pkl', 'rb') as f:
        A_inf_tmp = pickle.load(f)
    
    
    with open('../data/integral_spatial_passive_xd_' + str(xd) + '.pkl', 'rb') as f:
        A_inf_passive = pickle.load(f)
            
    title = r'$x_{d}=$' + f'{xd}'
    filename = '../results/A_inf_spatial_xd_'+ str(xd) + '.svg'
    plot_A_inf(A_inf_tmp, A_inf_passive, I_s_vec, I_d_vec, title, filename)
    
    return

def plot_spatial_dendrite_exp_input():
    
    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 21)
    
    with open('../data/integral_spatial_input_active.pkl', 'rb') as f:
        A_inf_tmp = pickle.load(f)
    
    with open('../data/integral_spatial_input_passive.pkl', 'rb') as f:
        A_inf_passive = pickle.load(f)
            
    title = 'Active Dendrite'
    filename = '../results/A_inf_exp_input_active.svg'
    plot_A_inf(A_inf_tmp, A_inf_passive, I_s_vec, I_d_vec, title, filename)
    
    title = 'Passive Dendrite'
    filename = '../results/A_inf_exp_input_passive.svg'
    plot_A_inf(A_inf_passive, A_inf_passive, I_s_vec, I_d_vec, title, filename)
    
    
    return

def plot_CV_versus_A_inf():
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    xd_I_list = [5, 8]
    
    for xd_I in xd_I_list:
        
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_active = pickle.load(f)
            
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_active = pickle.load(f)
        
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_passive = pickle.load(f)
            
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_passive = pickle.load(f)
                
        title = 'Active Dendrite' + ', $x_{d}=$' + f'{xd_I}'
        filename = '../results/integral_hotspot_active_xd_I_' + str(xd_I) + '.svg'

        
        plt.figure(figsize=(1.5, 1.5))
        alphas = [1, 0.7, 0.5]
        for i, I_d in enumerate(I_d_vec):
            plt.plot(A_inf_active[i, :], cv_active[i, :], label='$I_{d}=$' + f'{I_d}', color='tab:blue', alpha=alphas[i])
            
        plt.plot(A_inf_passive[i, :], cv_passive[0, :], '--', color='k', label='Passive')
        
        plt.xlim([0, 150])
        plt.xticks([0, 50, 100, 150])
        plt.xlabel(r'$A_{\infty}$ [Hz]')
        
        plt.yticks([0.5, 1.0, 1.5])
        plt.ylabel(r'$C_{v}$')
    
        plt.legend(frameon=False)
        sns.despine()
        plt.tight_layout()
        
        plt.savefig(f'../results/cv_versus_A_{xd_I}.svg')
    
    
    return

def plot_gauss_dendrite_gauss_input():
    """
    Plot ensemble firing rate for a population with a gaussian hotspot and gaussian input current along the dendrite.

    Returns
    -------
        Saves figures in the results folder.

    """
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    xd_I_list = [5, 8]
    
    for xd_I in xd_I_list:
        
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_active = pickle.load(f)
            
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_active = pickle.load(f)
        
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_passive = pickle.load(f)
            
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_passive = pickle.load(f)
                
        title = 'Active Dendrite' + ', $x_{d}=$' + f'{xd_I}'
        filename = '../results/integral_hotspot_active_xd_I_' + str(xd_I) + '.svg'
        plot_A_inf(A_inf_active, A_inf_passive, I_s_vec, I_d_vec, title, filename)
        
        filename = '../results/integral_hotspot_active_xd_I_' + str(xd_I) + '_cv.svg'
        plot_cv2(cv_active, cv_passive, I_s_vec, I_d_vec, title, filename)
        
        
        title = 'Passive Dendrite' + ', $x_{d}=$' + f'{xd_I}'
        filename = '../results/integral_hotspot_passive_xd_I_' + str(xd_I) + '.svg'
        plot_A_inf(A_inf_passive, A_inf_passive, I_s_vec, I_d_vec, title, filename)
        
        filename = '../results/integral_hotspot_passive_xd_I_' + str(xd_I) + '_cv.svg'
        plot_cv2(cv_passive, cv_passive, I_s_vec, I_d_vec, title, filename)
        
    return

def plot_gauss_dendrite_gauss_input_monte_carlo():
    """
    Plot ensemble firing rate for a population with a gaussian hotspot and gaussian input current along the dendrite.

    Returns
    -------
        Saves figures in the results folder.

    """
    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    xd_I_list = [7, 8, 9]
    
    for xd_I in xd_I_list:
        
        # Load A integral and monte carlo data for active case
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_active = pickle.load(f)
            
        with open('../data/monte_carlo_hotspot_active_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            results_active = pickle.load(f)
            A_inf_mc_data_active = results_active['A_inf_mc']
            A_inf_mc_error_active = results_active['error_mc']
        
        # Load Cv integral and monte carlo data for active case
        with open('../data/integral_hotspot_active_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_active = pickle.load(f)
            
        with open('../data/monte_carlo_hotspot_active_cv_xd_I_' + str(xd_I) + '.pkl',  'rb') as f:
            cv_active_mc = pickle.load(f)
                

        # Load A integral and monte carlo data for passive case
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            A_inf_passive = pickle.load(f)
            
        with open('../data/monte_carlo_hotspot_passive_xd_I_' + str(xd_I) + '.pkl', 'rb') as f:
            results_passive = pickle.load(f)
            A_inf_mc_data_passive = results_passive['A_inf_mc']
            A_inf_mc_error_passive = results_passive['error_mc']
        
        # Load Cv integral and monte carlo data for passive case
        with open('../data/integral_hotspot_passive_xd_I_' + str(xd_I) + '_cv.pkl', 'rb') as f:
            cv_passive = pickle.load(f)
            
        with open('../data/monte_carlo_hotspot_passive_cv_xd_I_' + str(xd_I) + '.pkl',  'rb') as f:
            cv_passive_mc = pickle.load(f)
             
        # Plot A active case with monte carlo data   
        title = 'Active Dendrite' + ', $x_{d}=$' + f'{xd_I}'
        filename = '../results/integral_hotspot_active_xd_I_' + str(xd_I) + '.svg'
        plot_A_inf_monte_carlo(A_inf_data=A_inf_active, A_inf_mc_data=A_inf_mc_data_active, A_inf_mc_error=A_inf_mc_error_active, title=title, filename=filename, A_inf_data_passive=A_inf_passive)
        
        # Plot Cv active case with monte carlo data 
        filename = '../results/integral_hotspot_active_xd_I_' + str(xd_I) + '_cv.svg'
        plot_cv(CV_data=cv_active, CV_mc_data=cv_active_mc, title=title, filename=filename, CV_data_passive=cv_passive)
        
        # Plot A passive case with monte carlo data 
        title = 'Passive Dendrite' + ', $x_{d}=$' + f'{xd_I}'
        filename = '../results/integral_hotspot_passive_xd_I_' + str(xd_I) + '.svg'
        plot_A_inf_monte_carlo(A_inf_data=A_inf_passive, A_inf_mc_data=A_inf_mc_data_passive, A_inf_mc_error=A_inf_mc_error_passive, title=title, filename=filename, A_inf_data_passive=A_inf_passive)
        
        # Plot Cv passive case with monte carlo data 
        filename = '../results/integral_hotspot_passive_xd_I_' + str(xd_I) + '_cv.svg'
        plot_cv(CV_data=cv_passive, CV_mc_data=cv_passive_mc, title=title, filename=filename, CV_data_passive=cv_passive)
        
    return


def plot_cv(CV_data, CV_mc_data, title, filename, CV_data_passive=None):

    I_d_vec = np.linspace(0, 5, 3)
    I_s_vec = np.linspace(0, 5, 31)
    
    I_d_mc_vec = np.linspace(0, 5, 3)
    I_s_mc_vec = np.linspace(0, 5, 5)
    
    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    plt.figure(figsize=(1.7, 1.5))
    
    for j, I_d in enumerate(I_d_vec):
        plt.plot(I_s_vec, CV_data[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    for j, I_d in enumerate(I_d_mc_vec):
        plt.plot(I_s_mc_vec, CV_mc_data[j, :],  '.', color='k')
    
        
    if CV_data_passive is not None: 
        plt.plot(I_s_vec, CV_data_passive[0, :], label='Passive', color='k', linestyle='--')
    
        
    
    plt.xlabel(r'$I_{s}$')
    plt.ylabel('CV')
    plt.xticks([0, 2.5, 5])
    plt.yticks([0.5, 1, 1.5])
    
    plt.xlim([-0.2, 5.2])
    
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig('../results/' + filename + '.svg')
    
    # save source image data in csv
    dic = {"CV_data" : CV_data, "CV_mc_data" : CV_mc_data, "CV_data_passive":CV_data_passive}
    dic["CV_data"] = [tuple(scores) for scores in dic["CV_data"]]
    dic["CV_data_passive"] = [tuple(scores) for scores in dic["CV_data_passive"]]
    dic["CV_mc_data"] = [tuple(scores) for scores in dic["CV_mc_data"]]
    data_frame = pd.DataFrame.from_records(dic)
    save_plot_data_csv(data_frame, filename='../data/' + filename + '_data.csv')
    
    return

def plot_cv_vs_A(A_active, A_mc_active, A_passive, A_mc_passive, cv_active, cv_passive, cv_mc_active, cv_mc_passive, title, filename):
    
    I_d_vec = np.linspace(0, 5, 3)
    
    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    plt.figure(figsize=(1.7, 1.5))
    
    # active
    #theory
    for j, I_d in enumerate(I_d_vec):
        plt.plot( A_active[j, :],cv_active[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        plt.plot(A_mc_active[j, :],cv_mc_active[j, :],  '.', color='k')
    
        
    plt.plot(A_passive[j, :], cv_passive[j, :], label='Passive', color='k', linestyle='--')
    plt.plot(A_mc_passive[j, :], cv_mc_passive[j, :], '.', color='k', label='Passive',)
    
        
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('CV')
    plt.xticks([0, 50, 100, 150])
    plt.yticks([0.5, 1, 1.5])
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig(filename)
    
    
    
    return

def plot_cv2(CV_data, CV_data_passive, I_s_vec, I_d_vec, title, filename):

    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    plt.figure(figsize=(1, 0.75))
    
    for j, I_d in enumerate(I_d_vec):
        plt.plot(I_s_vec, CV_data[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    
        
    if CV_data_passive is not None: 
        plt.plot(I_s_vec, CV_data_passive[0, :], label='Passive', color='k', linestyle='--')
    
        
    
    plt.xlabel(r'$I_{s}$')
    plt.ylabel(r'$C_{v}$')
    plt.xticks([0, 2.5, 5])
    plt.yticks([0.5, 1, 1.5])
    
    plt.xlim([-0.2, 5.2])
    
    
    plt.title(title)
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig(filename)
    
    return

def plot_cv_versus_input_location(CV_data, CV_data_passive, xd_vec, I_d_vec, title, filename):

    color = 'tab:blue'
    alpha = [1, 0.75, 0.5]
    
    plt.figure(figsize=(1, 0.75))
    
    for j, I_d in enumerate(I_d_vec):
        plt.plot(xd_vec, CV_data[j, :]-CV_data_passive[j, :], label=r'$I_{d}$='+f'{I_d}', color=color, alpha=alpha[j])
        
    
        
    #if CV_data_passive is not None: 
        #plt.plot(xd_vec, CV_data_passive[0, :], label='Passive', color='k', linestyle='--')
    
    #plt.vlines(8, 0, 2, color='tab:red', linestyle='--')
        
    
    plt.xlabel(r'$x_{d}$')
    plt.ylabel(r'$\Delta C_{v}$')
    plt.xticks([5, 10])
    plt.ylim([-0.1, 0.5])
    plt.yticks([0, 0.25, 0.5])
    plt.title(title)
    
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    
    plt.savefig(filename)
    
    return


def plot_figure1():
    """
    Fucntion to plot Figure 1b-d

    Returns:
        Figures saved in ../results/

    """
    
    # Figure 1A
    plot_input_potential(input_potential, filename='input_potential')
    
    
    # Figure 1B
    with open('../data/monte_carlo_passive.pkl',  'rb') as f:
        results_passive = pickle.load(f)
        
        A_inf_mc_data_passive = results_passive['A_inf_mc']
        #A_inf_mc_error_passive = results_passive['error_mc']
        
        
    with open('../data/integral_passive.pkl',  'rb') as f:
        A_inf_data_passive = pickle.load(f)
        
    
    with open('../data/monte_carlo_active.pkl',  'rb') as f:
        results_active = pickle.load(f)
        
        A_inf_mc_data_active = results_active['A_inf_mc']
        A_inf_mc_error_active = results_active['error_mc']
        
        
    with open('../data/integral_active.pkl',  'rb') as f:
        A_inf_data_active = pickle.load(f)
        
    filename = 'figure1_b'
    title = 'Active Dendrite'
    plot_A_inf_monte_carlo(A_inf_data_active, A_inf_mc_data_active, A_inf_mc_error_active, title, filename, A_inf_data_passive=A_inf_data_passive)
    
    
    # Figure 1C
    with open('../data/monte_carlo_passive_cv.pkl',  'rb') as f:
        cv_passive_mc = pickle.load(f)
        
    with open('../data/integral_passive_cv.pkl',  'rb') as f:
        cv_passive = pickle.load(f)
        
    #filename = '../results/figure1_passive_cv.svg'
    #title = 'Passive Dendrite'
    #plot_cv(cv_passive, cv_passive_mc, title, filename)
    
    
    with open('../data/monte_carlo_active_cv.pkl',  'rb') as f:
        cv_active_mc = pickle.load(f)
        
    with open('../data/integral_active_cv.pkl',  'rb') as f:
        cv_active = pickle.load(f)
        
    filename = 'figure1_c'
    title = 'Active Dendrite'
    plot_cv(cv_active, cv_active_mc, title, filename, cv_passive)
    
    # Figure 1D
    filename = '../results/figure1_cv_versus_A.svg'
    title = 'Active Dendrite'
    plot_cv_vs_A(A_active=A_inf_data_active, A_mc_active=A_inf_mc_data_active, A_passive=A_inf_data_passive, A_mc_passive=A_inf_mc_data_passive, cv_active=cv_active, cv_passive=cv_passive, cv_mc_active=cv_active_mc, cv_mc_passive=cv_passive_mc, title=title, filename=filename)
    
    
    
    return



def plot_supplementary_figure1():
    
    plot_input_potential(input_potential_gauss, filename='input_potential_gauss_8', xd_I=8)
    plot_input_potential(input_potential_gauss, filename='input_potential_gauss_5', xd_I=5)
    plot_gaussian_input(xd_I=8)
    plot_gaussian_filter()
    plot_effective_gauss_input_potential(input_potential_gauss, filename='effetive_input_potential_gauss_8', xd_I=8)
    plot_effective_gauss_input_potential(input_potential_gauss, filename='effetive_input_potential_gauss_5', xd_I=5)
    
    
    
    return


if __name__ == "__main__":
    
    # Figure 1
    plot_figure1()
    
    # Supplementary Figure 1
    plot_supplementary_figure1()
    
    
    
    

    
    #plot_figure_kernel_changes()
    
    # Figure 5
    #plot_CV_versus_A_inf()

    
