# Fig 1


- input_potential.csv contains data for the schematic in Fig. 1a

- figure1_bcd_inputs.csv contains the values of the inputs for the Fig. 1b-d
  - I_s_vec are the values of I_s for the theory curves
  - I_d_vec are the values of I_d for the theory
  - I_s_mc_vec are the values of I_s for the Monte Carlo simulations
  - I_d_mc_vec are the values of I_d for the Monte Carlo simulations


- figure1_b_data.csv contains values for the Fig. 1b
  - A_inf_data is the data for the theory curves with an active dendrite
  - A_inf_data_passive is the data for the theory curves with a passive dendrite
  - A_inf_mc_data are the estimates obtained by Monte Carlo simulations
  - Each row corresponds to a different value of dendritic input I_d_vec


- figure1_c_data.csv contains values for the Fig. 1c
  - Same as figure1_b_data expect for the values of CV instead of firing rate

- figure1_f.csv contains data for the f-I and CV-I curves obtained using the two-compartment LIF model

- figure1_h.csv contains data for the Fig. 1h

Note: Data for the traces in Fig. 1e and Fig. 1g is not provided because
we did not include a random seed when making the plots. Similar plots
can be generated using the code found in two_compartment_LIF.ipynb
