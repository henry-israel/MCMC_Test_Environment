'''
Interface class for MCMC, allows for likelihood settings etc.
'''

import numpy as np
from mcmc import mcmc_base, metropolis_hastings, jams_mcmc, hamiltonian_mcmc, jams_burner
from likelihood import likelihood_base, gaussian_likelihood, multi_modal_gaussian, likelihood_interface

class mcmc_interface:
    def __init__(self, mcmc_type, likelihood_type, **kwargs) -> None:

        kwargs_dict = {
            # Metropolis Hastings
            'step_sizes' : np.ones(2),
            
            # JAMS
            'alpha_arr' : np.ones(2),
            'beta' : 0.00001,
            'jump_epislon' : 0.1,
            'burn_in' : 10000,
            'burn_steps' : 10000,

            # Hamiltonian
            'time_step' : 0.1,
            'epsilon' : 0.1,
            'mass_matrix' : np.ones(2),
        }

        kwargs_dict.update(kwargs)

        self._likelihood = likelihood_interface(likelihood_type).get_likelihood()

        if mcmc_type == "metropolis_hastings":
            self._mcmc = metropolis_hastings()
            self._mcmc.likelihood = self._likelihood
            self._mcmc.set_step_sizes(kwargs_dict['step_sizes'])
            self._mcmc.current_state = self._likelihood.initial_state

        elif mcmc_type == "jams_mcmc_burnin":
            self._mcmc = jams_burner()
            self._mcmc.likelihood = self._likelihood
            self._mcmc.n_modes = self._likelihood.get_n_modes()
            
            self._mcmc.jump_epsilon = kwargs_dict['jump_epislon']
            self._mcmc.alphas = kwargs_dict['alpha_arr']
            self._mcmc.beta = kwargs_dict['beta']
            self._mcmc.nsteps_burn = kwargs_dict['burn_in']
            
        elif mcmc_type == "jams_mcmc":
            self._mcmc = jams_mcmc()
            self._mcmc.likelihood = self._likelihood

            self._mcmc.n_modes = self._likelihood.get_n_modes()

            self._mcmc.jump_epsilon = kwargs_dict['jump_epislon']
            self._mcmc.alphas = kwargs_dict['alpha_arr']
            self._mcmc.beta = kwargs_dict['beta']
            self._mcmc.current_state = self._likelihood.initial_state

        elif mcmc_type == "hamiltonian_mcmc":
            self._mcmc = hamiltonian_mcmc()
            self._mcmc.likelihood = self._likelihood

            self._mcmc.time_step = kwargs_dict['time_step']
            self._mcmc.epsilon = kwargs_dict['epsilon']
            self._mcmc.mass_matrix = kwargs_dict['mass_matrix']
            self._mcmc.current_state = self._likelihood.initial_state
            
        else:
            raise ValueError("MCMC type not recognized, must be metropolis_hastings, jams_mcmc_burnin, jams_mcmc, or hamiltonian_mcmc.")
        
    def __call__(self, nsteps: int) -> None:
        self._mcmc(nsteps)
