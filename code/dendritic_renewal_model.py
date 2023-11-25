"""Functions defining the dendritic renewal model and numerical integration."""
import numpy as np
from tqdm import trange
from scipy.special import erf


# FUNCTIONS DEFINING THE MODEL - parameters, inputs, hazard rates, survivor functions

def params_reset():
    """
    Generate a dictionary with the default neuron model parameters.

    Return
    ------
        params (dict): Parameters defining the neuron model.

    """
    params = {}
    
    # input potential parameters
    params['L'] = 10
    params['I_d'] = 1
    params['I_s'] = 1
    params['x_d'] = 8
    
    # dendritic rate parameters
    params['lambda_d'] = .1
    params['B'] = 1
    params['v_B'] = 2
    
    # somatic rate parameters
    params['lambda_s'] = .01
    params['R'] = 1
    params['delta_R'] = 1
    params['D'] = 2
    params['delta_D'] = 2
    
    params['tau'] = 10*1e-3 # hypothetical membrane time constant for scaling time.

    # gaussian mask parameters
    params['mu'] = 8
    params['sigma'] = 0.5
    
    # gaussian input parameters
    params['xd_I'] = 8
    params['sigma_I'] = 0.5
    
    return params

def input_potential_gauss(x_vec, params):
    
    L = params['L']
    I_d = params['I_d']
    I_s = params['I_s']
    
    a = np.exp(2*L)-1
    
    I_s_tilde = I_s*(np.exp(-x_vec) + (np.exp(x_vec)+np.exp(-x_vec))/a)

    xd = params['xd_I']
    sigma = params['sigma_I']
    
    gamma_pos = erf((xd-sigma**2)/(np.sqrt(2)*sigma)) - erf((xd - sigma**2 - L)/(np.sqrt(2)*sigma))
    gamma_neg = -1*(erf((-xd-sigma**2)/(np.sqrt(2)*sigma)) - erf((-xd - sigma**2 + L)/(np.sqrt(2)*sigma)))
    
    
    gamma_pos2 = erf((xd+sigma**2)/(np.sqrt(2)*sigma)) - erf((xd+sigma**2 - L)/(np.sqrt(2)*sigma))
    gamma_neg2 = -1*(erf((-xd+sigma**2)/(np.sqrt(2)*sigma)) - erf((-xd + sigma**2 + L)/(np.sqrt(2)*sigma)))
    
    A = (np.exp((1/2)*sigma**2))/(2*(np.exp(2*L) - 1))
    h = A*(gamma_pos*np.exp(x_vec-xd))
    h2 = A*(gamma_neg*np.exp(x_vec+xd))
    
    h3 = A*gamma_pos2*np.exp(-(x_vec-xd))
    h4 = A*gamma_neg2*np.exp(-(x_vec+xd))
    
    g1 =  erf((-sigma**2 - (x_vec-xd))/(np.sqrt(2)*sigma)) - erf((-L -sigma**2 + xd)/(np.sqrt(2)*sigma))
    g2 =  erf((sigma**2 + xd)/(np.sqrt(2)*sigma)) - erf((-(x_vec-xd -sigma**2))/(np.sqrt(2)*sigma))
    h5 = (1/2)*np.exp((1/2)*sigma**2)*(g1*np.exp(x_vec-xd) + g2*np.exp(-(x_vec-xd)))
    
    h6 = (1/2)*np.exp((1/2)*sigma**2 - (x_vec+xd))*(erf((L-xd+sigma**2)/(np.sqrt(2)*sigma))+erf((-xd+sigma**2)/(np.sqrt(2)*sigma)))
    
    
    return I_s_tilde + (I_d/2)*(h + h2 + h3 + h4 + h5 + h6)

def input_potential_gauss_0(params):
    
    L = params['L']
    I_d = params['I_d']
    I_s = params['I_s']
    x_vec = 0
    
    a = np.exp(2*L)-1
    
    I_s_tilde = I_s*(np.exp(-x_vec) + (np.exp(x_vec)+np.exp(-x_vec))/a)
 
    xd = params['xd_I']
    sigma = params['sigma_I']
    
    gamma_pos = erf((xd-sigma**2)/(np.sqrt(2)*sigma)) - erf((xd - sigma**2 - L)/(np.sqrt(2)*sigma))
    gamma_neg = -1*(erf((-xd-sigma**2)/(np.sqrt(2)*sigma)) - erf((-xd - sigma**2 + L)/(np.sqrt(2)*sigma)))
    
    
    gamma_pos2 = erf((xd+sigma**2)/(np.sqrt(2)*sigma)) - erf((xd+sigma**2 - L)/(np.sqrt(2)*sigma))
    gamma_neg2 = -1*(erf((-xd+sigma**2)/(np.sqrt(2)*sigma)) - erf((-xd + sigma**2 + L)/(np.sqrt(2)*sigma)))
    
    A = (np.exp((1/2)*sigma**2))/(2*(np.exp(2*L) - 1))
    h = A*(gamma_pos*np.exp(x_vec-xd))
    h2 = A*(gamma_neg*np.exp(x_vec+xd))
    
    h3 = A*gamma_pos2*np.exp(-(x_vec-xd))
    h4 = A*gamma_neg2*np.exp(-(x_vec+xd))
    
    g1 =  erf((-sigma**2 - (x_vec-xd))/(np.sqrt(2)*sigma)) - erf((-L -sigma**2 + xd)/(np.sqrt(2)*sigma))
    g2 =  erf((sigma**2 + xd)/(np.sqrt(2)*sigma)) - erf((-(x_vec-xd -sigma**2))/(np.sqrt(2)*sigma))
    h5 = (1/2)*np.exp((1/2)*sigma**2)*(g1*np.exp(x_vec-xd) + g2*np.exp(-(x_vec-xd)))
    
    h6 = (1/2)*np.exp((1/2)*sigma**2 - (x_vec+xd))*(erf((L-xd+sigma**2)/(np.sqrt(2)*sigma))+erf((-xd+sigma**2)/(np.sqrt(2)*sigma)))
    
    
    return I_s_tilde + (I_d/2)*(h + h2 + h3 + h4 + h5 + h6)

def input_potential(X, params):
    """
    Estimate the input potnential at locations X for a neuron with parameters given by params. Localized somatic and dendritic input.

    Args
    ----
        X (Ndarray): 1xN array of spatial locations along the dendrite.
        params (dict): Dicionary of parameter values defining the neuron model.

    Return
    ------
        Array of input potentials corresponing to the locations given by the input X.
    """
    I_s = params['I_s']
    I_d = params['I_d']
    x_d = params['x_d']
    L = params['L']
    
    N = 100 # approximation of infinite sum
    n = np.arange(-N, N+1) # n = {-N, -(N-1), ..., 0, ..., (N-1), N}
    
    if type(X)==int:  # if only a point estimate is needed at location X
        h = np.sum(I_s*np.exp(-1*np.abs(X-2*n*L)) + (I_d/2)*(np.exp(-1*np.abs(X-2*n*L-x_d)) + np.exp(-1*np.abs(X-2*n*L+x_d))))
    else: # get estimate of the input potential for each entry of X
        h = np.zeros(len(X))
        for i, x in enumerate(X): 
            h[i] = np.sum(I_s*np.exp(-1*np.abs(x-2*n*L)) + (I_d/2)*(np.exp(-1*np.abs(x-2*n*L-x_d)) + np.exp(-1*np.abs(x-2*n*L+x_d))))
        
    return h

def input_potential_0(params):
    """
    Get the input potnential at location x=0 for a neuron with parameters given by params. 
    For localized somatic and dendritic input.

    Args
    ----
        params (dict): Parameters defining the neuron model.

    Return
    ------
        Input potential at location x=0
    """    
    I_s = params['I_s']
    I_d = params['I_d']
    x_d = params['x_d']
    L = params['L']
    
    beta = (np.exp(2*L)+1)/(np.exp(2*L)-1)
    alpha = np.exp(-x_d) + (np.exp(-x_d) +np.exp(x_d))/(np.exp(2*L)-1)
    h = I_s*beta + I_d*alpha
    
    return h

def rate_soma0(t, t_hat, params, input_potential_0):
    """
    Calculate the initial somatic hazard rate.

    Args
    ----
        t (TYPE): Current time.
        t_hat (TYPE): Time of the last somatic spike.
        params (dict): Parameters defining the neuron model.

    Return
    ------
        rate (TYPE): DESCRIPTION.

    """
    lambda_s = params['lambda_s']
    R = params['R']
    delta_R = params['delta_R']
    
    h0 = input_potential_0(params)
    ref = -R*np.heaviside(delta_R - (t-t_hat), 1)
    rate = lambda_s*np.exp(h0 + ref)
    
    return rate

def rate_soma(t, x_p, t_p, t_hat, params, input_potential_0):
    """
    Calculate the soamtic hazard rate (after a dendritic spike has ocurred).

    Args
    ----
        t (TYPE): DESCRIPTION.
        x_p (TYPE): DESCRIPTION.
        t_p (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (dict): Parameters defining the neuron model.

    Return
    ------
        rate (TYPE): DESCRIPTION.

    """
    lambda_s = params['lambda_s']
    R = params['R']
    delta_R = params['delta_R']
    D = params['D']
    delta_D = params['delta_D']
    
    
    h0 = input_potential_0(params)
    ref = -R*np.heaviside(delta_R - (t-t_hat), 1)
    d_spike = D*np.heaviside(delta_D - (t-t_p), 1)
    rate = lambda_s*np.exp(h0 + ref + d_spike)
    
    return rate

def survivor_s0(t, t_hat, params, input_potential_0):
    """
    Calculate the initial somatic survivor function.

    Arg
    ----
        t (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.

    Return
    ------
        S (TYPE): DESCRIPTION.

    """
    delta_R = params['delta_R']
    R = params['R']
    lambda_s = params['lambda_s']
    
    h0 = input_potential_0(params)
    
    t_lower = t[t<delta_R]
    t_upper = t[t>=delta_R]

    S_lower = np.exp(-lambda_s*np.exp(h0 - R)*(t_lower-t_hat))
    S_upper = np.exp(-(lambda_s*(np.exp(h0-R)*delta_R + np.exp(h0)*(t_upper-t_hat-delta_R))))
    S = np.hstack((S_lower, S_upper))
    
    return S

def survivor_d(t, t_hat, params, input_potential):
    """
    Dendritic survivor probability.

    Args
    ----
        t (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.

    Returns
    -------
        S (TYPE): DESCRIPTION.

    """
    I_s = params['I_s']
    I_d = params['I_d']
    v_B = params['v_B']
    L = params['L']
    B = params['B']
    lambda_d = params['lambda_d']
    
    z = L/v_B 
    S = np.exp(-lambda_d*((I_s+I_d)*(t-t_hat) + B*((t-t_hat) - (t-t_hat-z)*np.heaviside(t-t_hat-z, 1))))

    return S

def survivor_d_spatial(t, t_hat, params, input_potential):
    """
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    t_hat : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    v_B = params['v_B']
    L = params['L']
    B = params['B']
    lambda_d = params['lambda_d']
    gamma = params['gamma']
    
    dx = 0.001
    x_grid = np.linspace(0, L, int(L/dx)+1)
    h = input_potential(x_grid, params)

    I_tilde = np.sum(h*np.exp(gamma*(x_grid-L)))*dx
    
    z = L/v_B 
    S = np.exp(-lambda_d*(I_tilde*(t-t_hat) + 
                          (B/gamma)*np.exp(-gamma*L)*(np.exp(gamma*v_B*((t-t_hat) - 
                                                                        (t-t_hat-z)*np.heaviside(t-t_hat-z, 1)))-1)))
    return S

def survivor_d_gauss(t, t_hat, params, input_potential):
    """
    Dendritic survivor function with a Gaussian mask applied to the dendritic hazard function.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    t_hat : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    v_B = params['v_B']
    L = params['L']
    B = params['B']
    lambda_d = params['lambda_d']
    mu = params['mu']
    sigma = params['sigma']
    
    dx = 0.001
    x_grid = np.linspace(0, L, int(L/dx)+1)
    h = input_potential(x_grid, params)

    I_tilde = np.sum(h*np.exp(-(x_grid-mu)**2 / (2*sigma**2)))*dx
    
    z = L/v_B 
    erf1 = erf((v_B*((t-t_hat) - (t-t_hat-z)*np.heaviside(t-t_hat-z, 1)) - mu)/(np.sqrt(2)*sigma))
    erf2 = erf(-mu / (np.sqrt(2)*sigma))
    
    
    S = np.exp(-lambda_d*(I_tilde*(t-t_hat) + ((B*np.sqrt(2*np.pi)*sigma)/(2*v_B))*(erf1 - erf2)))
                          
    return S

def survivor_function_estimate(intervals, T):
    """
    Estimate the survivor probability from the monte carlo data.

    Args
    ----
        intervals (list): Interspike intervals obtained by monte carlo sampling 
        T (Ndarray): Times to estimate the survivor function. 

    Return
    ------
        S (Ndarray): Survivor probability as a function of input T.  

    """
    intervals = np.asarray(intervals)
    
    S = np.zeros(len(T))
    for i, t in enumerate(T): 
        S[i] = np.sum(intervals>t)/len(intervals)
        
    return S

def dendritic_rate(t, t_hat, params):
    """
    Integral of the dendritic hazard rate over space. Non-spatial case.

    Args
    ----
        t (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.

    Return
    ------
        continuous_part (TYPE): DESCRIPTION.
        continuous_part_vec (TYPE): DESCRIPTION.
        discrete_part (TYPE): DESCRIPTION.

    """
    lambda_d = params['lambda_d']
    L = params['L']
    v_B = params['v_B']
    B = params['B']    
    h = params['h']
    dx = params['dx']
    
    continuous_part  = lambda_d*np.sum(h)*dx
    discrete_part = lambda_d*B*np.heaviside(L/v_B - (t-t_hat), 1)
    continuous_part_vec = lambda_d*h
    
    return continuous_part, continuous_part_vec, discrete_part

def dendritic_rate_spatial(t, t_hat, params):
    """
    Integral of the dendritic hazard rate over space. Exponential filter case.

    Args
    ----
        t (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.

    Return
    ------
        continuous_part (TYPE): DESCRIPTION.
        continuous_part_vec (TYPE): DESCRIPTION.
        discrete_part (TYPE): DESCRIPTION.

    """   
    lambda_d = params['lambda_d']
    L = params['L']
    v_B = params['v_B']
    B = params['B']
    h = params['h']
    dx = params['dx']
    gamma = params['gamma']
    
    x_grid = params['x_grid']
    
    continuous_part  = lambda_d*np.sum(h*np.exp(gamma*(x_grid-L)))*dx
    discrete_part = lambda_d*B*np.exp(gamma*(v_B*(t-t_hat)- L))*np.heaviside(L/v_B - (t-t_hat), 1)
    
    continuous_part_vec = lambda_d*h*np.exp(gamma*(x_grid-L))
    
    return continuous_part, continuous_part_vec, discrete_part

def dendritic_rate_gauss(t, t_hat, params):
    """
    Integral of the dendritic hazard rate over space. Gaussian filter case.

    Args
    ----
        t (TYPE): DESCRIPTION.
        t_hat (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.

    Return
    ------
        continuous_part (TYPE): DESCRIPTION.
        continuous_part_vec (TYPE): DESCRIPTION.
        discrete_part (TYPE): DESCRIPTION.

    """   
    lambda_d = params['lambda_d']
    L = params['L']
    v_B = params['v_B']
    B = params['B']
    h = params['h']
    dx = params['dx']
    mu = params['mu']
    sigma = params['sigma']
    
    x_grid = params['x_grid']
    
    gauss = np.exp(-(x_grid-mu)**2 / (2*sigma**2)) 
    continuous_part  = lambda_d*np.sum(h*gauss)*dx    
    discrete_part = lambda_d*B*np.exp(-(v_B*(t-t_hat)-mu)**2 / (2*sigma**2))*np.heaviside(L/v_B - (t-t_hat), 1)

    
    continuous_part_vec = lambda_d*h*gauss
    
    return continuous_part, continuous_part_vec, discrete_part


#####################################################################################################
###################### FUNCTIONS FOR SIMULATING THE NEURON RESPONSE PROPERTIES ######################
#####################################################################################################

###################### MONTE CARLO SIMULATION FUNCTIONS ######################
def monte_carlo(nb_steps, dt, params, dendritic_rate, input_potential_0, input_potential):
    """
    Monte Carlo sample the dendritic renewal model.
    
    Args
    ----
        nb_steps (TYPE): DESCRIPTION.
        dt (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.
        dendritic_rate (TYPE): DESCRIPTION.

    Return
    ------
        intervals (list): Somatic interspike intervals. 
        somatic_spike_train (Ndarray): Array of spikes (0's and 1's)
        dendritic_spike_train (Ndarray): Array of spikes (0's and 1's)
        x_d (list): Locations of the dendritic spikes along the dendrite.

    """
    L = params['L']
    v_B = params['v_B']
    dx = v_B*dt
    x_grid = np.linspace(0, L, int(L/dx)+1)
    
    intervals = []
    x_d = [0] # This isn't really used here, but we init it to start with a value of 0. 
    last_s_spike = 0 
    last_d_spike = 0
    somatic_spike_train = np.zeros(nb_steps)
    dendritic_spike_train = np.zeros(nb_steps)

    h = input_potential(x_grid, params) 
    params['h'] = h
    params['dx'] =  dx
    params['x_grid'] = x_grid
    
    dendrite_alive = True
    for i in range(nb_steps):
        
        if dendrite_alive==True: 

            continuous_part, continuous_part_vec, discrete_part = dendritic_rate(i*dt, last_s_spike*dt, params)
            rand_d = np.random.rand()
            prob_d = 1-np.exp(-(continuous_part+discrete_part)*dt)
            if rand_d <= prob_d: # check to see if the dendrite fires in this time bin

                x_loc = False
                r1 = np.random.rand()
                if r1<=(discrete_part/(continuous_part+discrete_part)): # Check if dendritic spike occurs at the BAP location
                    x_d.append(v_B*(i-last_s_spike)*dt) # save BAP location as location of dendritic spike
                    x_loc=True
                else:
                    while x_loc==False:
                        x_rand = np.random.randint(0, len(x_grid))

                        r2 = np.random.rand()
                        if r2 <= continuous_part_vec[x_rand]/(continuous_part+discrete_part): # Check where the dendritic spike occured
                            x_d.append(x_grid[x_rand])
                            x_loc = True

                last_d_spike = i
                dendrite_alive=False
                dendritic_spike_train[i] = 1
                continue
        
        # Initial somatic hazard function
        if dendrite_alive==True: # check to see if the soma fires
            rate_s0_tmp = rate_soma0(i*dt, last_s_spike*dt, params, input_potential_0)
            rand_s = np.random.rand()
            prob_s = 1-np.exp(-rate_s0_tmp*dt)
            if rand_s <= prob_s: 
                intervals.append((i-last_s_spike)*dt)
                last_s_spike = i
                somatic_spike_train[i] = 1
                continue
        
        # Somatic hazard function (after dendritic spike has ocurred)
        if dendrite_alive==False: 
            rate_s_tmp = rate_soma(i*dt, x_d[-1], last_d_spike*dt, last_s_spike*dt, params, input_potential_0)
            rand_s = np.random.rand()
            prob_s = 1-np.exp(-rate_s_tmp*dt)
            if rand_s <= prob_s: 
                intervals.append((i-last_s_spike)*dt)
                last_s_spike = i
                somatic_spike_train[i] = 1
                dendrite_alive=True
                continue

    return intervals, somatic_spike_train, dendritic_spike_train, x_d

def simulate_population_activity(nb_neurons, params, monte_carlo_fn, dendritic_rate, input_potential_0, input_potential):
    """
    Estimate the population activity from a population of monte carlo sampled neurons.

    Args
    ----
        nb_neurons (int): Number of neurons in the population. 
        params (dict): Parameters that define the neuron model.
        monte_carlo_fn (func): Monte carlo function to be sampled
        dendritic_rate (func): Dendritic hazard rate function. Can be spatial or non-spatial. 

    Return
    ------
        A (numpy array): Population activity as a function of time.

    """
    nb_steps = 3000
    dt = 0.01
    
    spike_trains_all = []
    for i in trange(nb_neurons):
        intervals, spike_train, _, _ = monte_carlo_fn(nb_steps, dt, params, dendritic_rate, input_potential_0, input_potential)
        spike_trains_all.append(spike_train)
        
    spike_train_all = np.asarray(spike_trains_all)
    
    A = np.mean(spike_train_all, axis=0)/dt
    
    return A

def coef_variation(intervals):
    """
    Calculate the coefficient of varaition

    Args:
        intervals (numpy array): Inter-spike-intervals

    Returns:
        TYPE: Coefficient of variation

    """
    
    mean = np.mean(intervals)
    std = np.std(intervals)
    
    return std/mean

###################### INTEGRATION METHOD FUNCTIONS ######################
def integrate_survivor(t_max, params, epsilon, dendritic_rate, survivor_d, input_potential_0, input_potential):
    """
    Numerically integrate the survivor functions.

    Args
    ----
        t_max (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.
        epsilon (TYPE): DESCRIPTION.
        dendritic_rate (func): Integral of the dendritic hazard rate over space. Can we for the spatial or non-spatial case. 
        survivor_d (func): Survivor function of the dendrite. Can be for the spatial or non-spatial case.

    Return
    ------
        S_theory TYPE: Marginal somatic survivor probability as a function of time
        t_prime_vec TYPE: The times corresponding to each value in S_theory

    """
    t_hat = 0
    dtp = t_max/10000
    dxp = 0.01
    x_p = 0 # Currently not used in the somatic survivor function, but we pass it as an input anyway for now.

    L = params['L']

    
    t_prime_vec = np.linspace(0, t_max, int(t_max/dtp)+1)
    x_prime_vec = np.linspace(0, L, int(L/dxp)+1)
    
    h = input_potential(x_prime_vec, params)
    params['h'] = h
    params['dx'] =  dxp
    params['x_grid'] = x_prime_vec
    
    S_theory = survivor_s0(t_prime_vec, t_hat, params, input_potential_0)*survivor_d(t_prime_vec, t_hat, params, input_potential)
    
    tmp1 = survivor_d(t_prime_vec, t_hat, params, input_potential)*survivor_s0(t_prime_vec, t_hat, params, input_potential_0)
    continuous_part, _, discrete_part = dendritic_rate(t_prime_vec, t_hat, params)
    rate_dt = continuous_part + discrete_part
    
    # Get matrix of the somatic survivor probability. Did this to make numerical integration faster.
    t_pp, tt = np.meshgrid(t_prime_vec, t_prime_vec)
    rate_s_mat = rate_soma(tt, x_p, t_pp, t_hat, params, input_potential_0)
    rate_s_mat_lower = np.tril(rate_s_mat)
    S_mat = np.exp(-np.cumsum(rate_s_mat_lower, axis=0)*dtp)
    
    for i, t in enumerate(t_prime_vec):    
        S_theory[i] += np.sum(tmp1[:i]*rate_dt[:i]*S_mat[i, :i])*dtp
        
        if S_theory[i]<epsilon: # If changes in S are small stop the numerical integration
            return S_theory[:i], t_prime_vec[:i]

    return S_theory, t_prime_vec

def inter_spike_interval_density(t_max, params, rate_soma0, rate_soma, dendritic_rate, survivor_d, input_potential_0, input_potential):
    """
    Numerically evaluate the somatic ISI distribution.

    Args:
        t_max (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.
        rate_soma0 (TYPE): DESCRIPTION.
        rate_soma (TYPE): DESCRIPTION.
        dendritic_rate (TYPE): DESCRIPTION.
        survivor_d (TYPE): DESCRIPTION.
        input_potential_0 (TYPE): DESCRIPTION.
        input_potential (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    t_hat = 0
    dtp = t_max/10000
    dxp = 0.01
    x_p = 0 # Currently not used in the somatic survivor function, but we pass it as an input anyway for now.

    L = params['L']

    
    t_prime_vec = np.linspace(0, t_max, int(t_max/dtp)+1)
    x_prime_vec = np.linspace(0, L, int(L/dxp)+1)
    
    h = input_potential(x_prime_vec, params)
    params['h'] = h
    params['dx'] =  dxp
    params['x_grid'] = x_prime_vec
    
    S_theory = rate_soma0(t_prime_vec, t_hat, params, input_potential_0)*survivor_s0(t_prime_vec, t_hat, params, input_potential_0)*survivor_d(t_prime_vec, t_hat, params, input_potential)
    
    tmp1 = survivor_d(t_prime_vec, t_hat, params, input_potential)*survivor_s0(t_prime_vec, t_hat, params, input_potential_0)
    continuous_part, _, discrete_part = dendritic_rate(t_prime_vec, t_hat, params)
    rate_dt = continuous_part + discrete_part
    
    # Get matrix of the somatic survivor probability. Did this to make numerical integration faster.
    t_pp, tt = np.meshgrid(t_prime_vec, t_prime_vec)
    rate_s_mat = rate_soma(tt, x_p, t_pp, t_hat, params, input_potential_0)
    rate_s_mat_lower = np.tril(rate_s_mat)
    S_mat = np.exp(-np.cumsum(rate_s_mat_lower, axis=0)*dtp)
    
    S_mat = S_mat*rate_s_mat_lower
    
    for i, t in enumerate(t_prime_vec):    
        S_theory[i] += np.sum(tmp1[:i]*rate_dt[:i]*S_mat[i, :i])*dtp

    return S_theory, t_prime_vec

def get_A_inf(I_s_vec, I_d_vec, params, dendritic_rate, survivor_dendrite, input_potential_0, input_potential):
    """
    Calucluate the stationary firing rate using the integration method.

    Args
    ----
        I_s_vec (numpy array): Somatic input currents
        I_d_vec (numpy array): Dendritic input currents
        params (dict): Neuron model parameters
        dendritic_rate (func): Dendritic hazard rate function
        survivor_dendrite (func): Dendritic survivor function

    Return
    ------
        A_inf_mat (numpy array): Firing rates for all pairs of somatic and dendritic inputs

    """
    epsilon = 0.001
    tau = params['tau']
    A_inf_mat = np.zeros((len(I_d_vec), len(I_s_vec)))
    for j, I_d in enumerate(I_d_vec):
        params['I_d'] = I_d
        for i, I_s in enumerate(I_s_vec):
            params['I_s'] = I_s
            if I_s == 0:
                t_max = 500
            else:
                t_max = np.log(1/0.01)/(params['lambda_s']*params['I_s']) # estimate the value of t_max from initial survivor probability
            
            S_theory, t_vec_theory = integrate_survivor(t_max, params, epsilon, dendritic_rate, survivor_dendrite, input_potential_0, input_potential)
            A_inf_mat[j, i] = (1/(np.sum(S_theory)*t_vec_theory[1]*tau))

    return A_inf_mat

def get_ensemble_response(I_s_vec, I_d_vec, params, dendritic_rate, survivor_dendrite, input_potential_0, input_potential):
    """
    Calucluate the stationary firing rate and cv using the integration method.

    Args
    ----
        I_s_vec (TYPE): DESCRIPTION.
        I_d_vec (TYPE): DESCRIPTION.
        params (TYPE): DESCRIPTION.
        dendritic_rate (TYPE): DESCRIPTION.
        survivor_dendrite (TYPE): DESCRIPTION.

    Return
    ------
        A_inf_mat (TYPE): DESCRIPTION.

    """
    epsilon = 0.001
    tau = params['tau']
    A_inf_mat = np.zeros((len(I_d_vec), len(I_s_vec)))
    cv_mat = np.zeros((len(I_d_vec), len(I_s_vec)))
    for j, I_d in enumerate(I_d_vec):
        params['I_d'] = I_d
        for i, I_s in enumerate(I_s_vec):
            params['I_s'] = I_s
            if I_s == 0:
                t_max = 500
            else:
                t_max = np.log(1/0.01)/(params['lambda_s']*params['I_s']) # estimate the value of t_max from initial survivor probability
            
            S_theory, t_vec_theory = integrate_survivor(t_max, params, epsilon, dendritic_rate, survivor_dendrite, input_potential_0, input_potential)
            A_inf_mat[j, i] = (1/(np.sum(S_theory)*t_vec_theory[1]*tau))    
            
            T_first_moment = (np.sum(S_theory)*t_vec_theory[1])
            T_second_moment = 2*np.sum(S_theory*t_vec_theory)*t_vec_theory[1]
            cv_mat[j, i] = np.sqrt((T_second_moment - T_first_moment**2) / T_first_moment**2)
    
    
    return A_inf_mat, cv_mat

def get_cv(I_s_vec, I_d_vec, params, dendritic_rate, survivor_dendrite, input_potential_0, input_potential):
    
    epsilon = 0.001
    cv_mat = np.zeros((len(I_d_vec), len(I_s_vec)))
    for j, I_d in enumerate(I_d_vec):
        params['I_d'] = I_d
        for i, I_s in enumerate(I_s_vec):
            params['I_s'] = I_s
            if I_s == 0:
                t_max = 500
            else:
                t_max = np.log(1/0.01)/(params['lambda_s']*params['I_s']) # estimate the value of t_max from initial survivor probability
    
            S_theory, t_vec_theory = integrate_survivor(t_max, params, epsilon, dendritic_rate, survivor_dendrite, input_potential_0, input_potential)
    
            T_first_moment = (np.sum(S_theory)*t_vec_theory[1])
            T_second_moment = 2*np.sum(S_theory*t_vec_theory)*t_vec_theory[1]
    
            cv_mat[j, i] = (T_second_moment - T_first_moment**2) / T_first_moment**2
    
    return np.sqrt(cv_mat)
