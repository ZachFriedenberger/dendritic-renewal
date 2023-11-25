import numpy as np
import pickle

from dendritic_renewal_model import simulate_population_activity, params_reset, dendritic_rate, get_A_inf, survivor_d, monte_carlo, input_potential, input_potential_0, coef_variation, get_cv
from dendritic_renewal_model import input_potential_gauss_0, input_potential_gauss, dendritic_rate_gauss, survivor_d_gauss, get_ensemble_response

######################### FUNCTIONS FOR GENERATING FIGURE 1 #########################

def get_integral_method_data(params, I_s_vec, I_d_vec, dendritic_rate, survivor_d, input_potential_0, input_potential, filename):
    """
    Function that calls get_A_inf and saves the output firing rates. get_A_inf calcualtes the firing rate

    Args:
        params (dict): Neuron model parameters
        I_s_vec (numpy array): Somatic input currents 
        I_d_vec (numpy array): Dendritic input currents
        dendritic_rate (func): Dendritic hazard rate function
        survivor_d (func): Dendritic survivor function
        input_potential_0 (func): Input potential at the soma
        input_potential (func): Input potential along the dendrite
        filename (str): file name for pickled output data
    Returns:
        Saves pickled data in ..data/filename.pkl

    """

    
    A_inf_tmp = get_A_inf(I_s_vec, I_d_vec, params, dendritic_rate, survivor_d, input_potential_0, input_potential)
    
    with open('../data/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(A_inf_tmp, f)
    
    return

def get_integral_method_cv(params, I_s_vec, I_d_vec, dendritic_rate, survivor_d, input_potential_0, input_potential, filename):
    """
    Function that calls get_cv and saves the output CVs. get_cv calcualtes the CV.

    Args:
        params (dict): Neuron model parameters
        I_s_vec (numpy array): Somatic input currents 
        I_d_vec (numpy array): Dendritic input currents
        dendritic_rate (func): Dendritic hazard rate function
        survivor_d (func): Dendritic survivor function
        input_potential_0 (func): Input potential at the soma
        input_potential (func): Input potential along the dendrite
        filename (str): file name for pickled output data
    Returns:
        Saves pickled data in ..data/filename.pkl

    """

    cv_mat = get_cv(I_s_vec, I_d_vec, params, dendritic_rate, survivor_d, input_potential_0, input_potential)
    
    with open('../data/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(cv_mat, f)
    
    return

def get_monte_carlo_data(params, monte_carlo_fn, dendritic_rate_fn, input_potential_0, input_potential, filename):
    """
    Monte Carlo sample dendritic renewal model and use intervals to calculate the firing rate

    Args:
        params (TYPE): DESCRIPTION.
        monte_carlo_fn (func): Monte carlo sampler
        dendritic_rate_fn (func): Dendritic hazard rate
        input_potential_0 (func): Input potential at the soma
        input_potential (func): INput potential along the dendrite
        filename (str): file name for saving output

    Returns:
        Saves firing rates in pickled file at ../data/filename.pkl

    """
    I_d_mc_vec = np.linspace(0, 5, 3)
    I_s_mc_vec = np.linspace(0, 5, 5)    
    tau = params['tau']
    
    nb_neurons = 10000
    A_inf_mc = np.zeros((len(I_d_mc_vec), len(I_s_mc_vec)))
    error_mc = np.zeros((len(I_d_mc_vec), len(I_s_mc_vec)))
    
    for j, I_d in enumerate(I_d_mc_vec):
        params['I_d'] = I_d
        for i, I_s in enumerate(I_s_mc_vec):
            params['I_s'] = I_s
    
            A_tmp = simulate_population_activity(nb_neurons, params, monte_carlo_fn, dendritic_rate_fn, input_potential_0, input_potential)
            
            A_tmp_len = len(A_tmp)
            last_quarter = A_tmp_len - int(A_tmp_len/4)
            
            A_inf_mc[j, i] = np.mean(A_tmp[last_quarter:])
            error_mc[j, i] = np.std(A_tmp[last_quarter:])/np.sqrt(int(A_tmp_len/4))
    
    
    results = {'A_inf_mc': A_inf_mc/tau, 'error_mc': error_mc/tau}

    # save data to file for plotting
    with open('../data/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    
    return

def get_monte_carlo_cv(params, monte_carlo_fn, dendritic_rate_fn, input_potential_0, input_potential, filename):
    """
    Monte Carlo sample dendritic renewal model and use intervals for calculating the CV.

    Args:
        params (TYPE): DESCRIPTION.
        monte_carlo_fn (func): Monte carlo sampler
        dendritic_rate_fn (func): Dendritic hazard rate
        input_potential_0 (func): Input potential at the soma
        input_potential (func): INput potential along the dendrite
        filename (str): file name for saving output

    Returns:
        Saves firing rates in pickled file at ../data/filename.pkl

    """
    
    I_d_mc_vec = np.linspace(0, 5, 3)
    I_s_mc_vec = np.linspace(0, 5, 5)    
    nb_steps = 10**6
    cv_mc = np.zeros((len(I_d_mc_vec), len(I_s_mc_vec)))
    
    for j, I_d in enumerate(I_d_mc_vec):
        params['I_d'] = I_d
        for i, I_s in enumerate(I_s_mc_vec):
            params['I_s'] = I_s
            
            # Monte carlo takes a long time to get spikes at low inputs, so I have incrementally increased the step size to speed up the simulation.
            if I_s<=1.25: 
                dt = 1
            if I_s>1.25 and I_s<=2.5:
                dt = 0.1
            if I_s>2.5:
                dt = 0.01
    
            intervals, _, _, _ = monte_carlo_fn(nb_steps, dt, params, dendritic_rate_fn, input_potential_0, input_potential)
            
            half = int(len(intervals)/2)
            cv_mc[j, i] = coef_variation(intervals[half:])
    
    
    # save data to file for plotting
    with open('../data/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(cv_mc, f)
    
    
    return

def get_dspike_kernel_changes():
    """
    Loop over different dendritic spike kernel shapes

    Returns:
        Saves data

    """
    
    D_vec = np.linspace(0, 5, 30)
    delta_D_vec = np.linspace(1, 15, 30)
    get_dspike_kernel_change_data(param1_vec=D_vec, param1_key='D', param2_vec=delta_D_vec, param2_key='delta_D')
    
    return

def get_dspike_kernel_change_data(param1_vec, param1_key, param2_vec, param2_key):
    """
    
    Get firing rate and cv using the integration method while changing the dendritic spike shape

    Args:
        param1_vec (numpy array): DESCRIPTION.
        param1_key (str): DESCRIPTION.
        param2_vec (numpy array): DESCRIPTION.
        param2_key (str: DESCRIPTION.

    Returns:
        Data is save in ..data/kernelchanges2/

    """

    I_d_vec = [2.5]
    I_s_vec = np.linspace(0, 5, 31)
    
    
    for param1 in param1_vec:
        for param2 in param2_vec:
            
            params = params_reset()
            params[param1_key] = param1
            params[param2_key] = param2
            
            A_inf_tmp, cv_tmp = get_ensemble_response(I_s_vec=I_s_vec, I_d_vec=I_d_vec, params=params, dendritic_rate=dendritic_rate, survivor_dendrite=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential)
           
            
            with open('../data/kernel_change_data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'wb') as f:
                pickle.dump(A_inf_tmp, f)
                
            with open('../data/kernel_change_data/cv_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change.pkl', 'wb') as f:
                    pickle.dump(cv_tmp, f)
            
            params['D'] = 0
            A_inf_passive, cv_passive = get_ensemble_response(I_s_vec=I_s_vec, I_d_vec=I_d_vec, params=params, dendritic_rate=dendritic_rate, survivor_dendrite=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential)
            
            
            with open('../data/kernel_change_data/A_inf_' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'wb') as f:
                pickle.dump(A_inf_passive, f)
                
            with open('../data/kernel_change_data/cv' + param1_key + '_' + str(param1) + param2_key + '_' + str(param2) + '_change_passive.pkl', 'wb') as f:
                    pickle.dump(cv_passive, f)
            
    
    return

def figure1_data():
    """
    Generates data used for plotting Figures b-d and f-h

    Returns:
        Pickled data saved in a subdirectory called data
    """
    
    ###############################################################
    ################ Generate data for Figure 1b-d ################
    ###############################################################
    
    I_d_vec = np.linspace(0, 5, 3) # currents to run analysis over
    I_s_vec = np.linspace(0, 5, 31)
    
    print('Running Figure 1b-d data: passive dendrite')
    ################ Passive dendrite ################ 
    params = params_reset()
    params['D'] = 0
    
    # Firing rate and CV using the integration method
    get_integral_method_data(params=params, I_s_vec=I_s_vec, I_d_vec=I_d_vec, dendritic_rate=dendritic_rate, survivor_d=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential, filename='integral_passive')

    get_integral_method_cv(params=params, I_s_vec=I_s_vec, I_d_vec=I_d_vec, dendritic_rate=dendritic_rate, survivor_d=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential, filename='integral_passive_cv')
    
    # Firing rate and CV using Monte Carlo
    get_monte_carlo_data(params=params, monte_carlo_fn=monte_carlo, dendritic_rate_fn=dendritic_rate, input_potential_0=input_potential_0, input_potential=input_potential, filename='monte_carlo_passive')

    get_monte_carlo_cv(params=params, monte_carlo_fn=monte_carlo, dendritic_rate_fn=dendritic_rate, input_potential_0=input_potential_0, input_potential=input_potential, filename='monte_carlo_passive_cv')

    print('Running Figure 1b-d data: active dendrite')
    ################ Active dendrite ################
    params = params_reset()
    params['D'] = 2
    
    get_integral_method_data(params=params, I_s_vec=I_s_vec, I_d_vec=I_d_vec, dendritic_rate=dendritic_rate, survivor_d=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential, filename='integral_active')

    get_integral_method_cv(params=params, I_s_vec=I_s_vec, I_d_vec=I_d_vec, dendritic_rate=dendritic_rate, survivor_d=survivor_d, input_potential_0=input_potential_0, input_potential=input_potential, filename='integral_active_cv')
    
    get_monte_carlo_data(params=params, monte_carlo_fn=monte_carlo, dendritic_rate_fn=dendritic_rate, input_potential_0=input_potential_0, input_potential=input_potential, filename='monte_carlo_active')

    get_monte_carlo_cv(params=params, monte_carlo_fn=monte_carlo, dendritic_rate_fn=dendritic_rate, input_potential_0=input_potential_0, input_potential=input_potential, filename='monte_carlo_active_cv')
    
    ###############################################################
    ################ Generate data for Figure 1f-h ################
    ###############################################################
    print('Running Figure 1f-h data: dendritic kernel changes')
    get_dspike_kernel_changes()

    return

######################### FUNCTIONS FOR GENERATING SUPPLEMENTARY FIGURE 1 #########################

def get_integral_method_spatial_data(params, dendritic_rate, survivor_d, input_potential_0, input_potential, filename):
    """
    Get firing rate and cv for spatial inputs and hotspots

    Args:
        params (TYPE): DESCRIPTION.
        dendritic_rate (TYPE): DESCRIPTION.
        survivor_d (TYPE): DESCRIPTION.
        input_potential_0 (TYPE): DESCRIPTION.
        input_potential (TYPE): DESCRIPTION.
        filename (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    
    I_d_vec = np.linspace(0, 10, 3)
    I_s_vec = np.linspace(0, 5, 31)
    

    A_inf_tmp = get_A_inf(I_s_vec, I_d_vec, params, dendritic_rate, survivor_d, input_potential_0, input_potential)
    
    cv_tmp = get_cv(I_s_vec, I_d_vec, params, dendritic_rate, survivor_d, input_potential_0, input_potential)
    
    with open('../data/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(A_inf_tmp, f)
        
    with open('../data/'+ filename + '_cv.pkl', 'wb') as f:
        pickle.dump(cv_tmp, f)
    
    return

def get_dendritic_hotspot_data():
    """
    Generate data for model with Gaussian dendritic hotspots and Gaussian inputs

    Returns:
        Pickled data in ../data/

    """
    
    params = params_reset()
    xd_I_list = [5, 8]
    
    for xd_I in xd_I_list:
    
        params['xd_I'] = xd_I
        params['D'] = 2
        
        get_integral_method_spatial_data(params=params, dendritic_rate=dendritic_rate_gauss, survivor_d=survivor_d_gauss, input_potential_0=input_potential_gauss_0, input_potential=input_potential_gauss, filename='integral_hotspot_active_xd_I_' + str(xd_I))
        
        params['D'] = 0
        get_integral_method_spatial_data(params=params, dendritic_rate=dendritic_rate_gauss, survivor_d=survivor_d_gauss, input_potential_0=input_potential_gauss_0, input_potential=input_potential_gauss, filename='integral_hotspot_passive_xd_I_' + str(xd_I))
        
         
    return

def supplmentary_figure1_data():
    """
    Generate data for supplemental figure 1 - Gaussian hotspot

    Returns:

    """
    print('Running Figure S1 data: dendritic hotspots')
    get_dendritic_hotspot_data()
    

    return

if __name__ == "__main__":

    figure1_data()
    supplmentary_figure1_data()
