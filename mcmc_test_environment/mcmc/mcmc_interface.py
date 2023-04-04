'''
Interface class for MCMC, allows for likelihood settings etc.
'''

import numpy as np
from .metropolis_hastings import metropolis_hastings, jams_mcmc, jams_mcmc_burner, adaptive_metropolis_hastings, metropolis_with_explicit_jump
from .hamiltonian import hamiltonian_mcmc
from ..likelihood.likelihood_zoo import likelihood_interface
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

class mcmc_interface:
    def __init__(self, mcmc_type, likelihood_type, **kwargs) -> None:

        kwargs_dict = {
            # Metropolis Hastings
            'step_sizes' : np.ones(2),
            'throw_matrix' : np.diag(np.ones(2)),
            # JAMS
            'alpha_arr' : np.ones(2),
            'beta' : 0.00001,
            'jump_epsilon' : 0.1,
            'burn_in' : 10000,
            'burn_steps' : 10000,

            # Hamiltonian
            'time_step' : 0.1,
            'epsilon' : 0.1,
            'mass_matrix' : np.ones(2),

            # Likelihood Settings
            'covariance' : np.ones((2,2)),
            'mu' : np.ones(2)
        }

        kwargs_dict.update(kwargs)

        self._likelihood = likelihood_interface(likelihood_type).get_likelihood()
        self._likelihood.set_likelihood_function(mu=kwargs_dict['mu'], covariance=kwargs_dict['covariance'])
        self._mcmc = None

        self._mcmc_type = mcmc_type

        if mcmc_type == "metropolis_hastings" or mcmc_type=="adaptive_metropolis_hastings" or mcmc_type=="jump_metropolis":
            if(mcmc_type == "metropolis_hastings"):
                self._mcmc = metropolis_hastings.metropolis_hastings()
            elif mcmc_type=="adaptive_metropolis_hastings":
                self._mcmc = adaptive_metropolis_hastings.adaptive_metropolis_hastings()
            else:
                self._mcmc = metropolis_with_explicit_jump.metropolis_with_explicit_jump()

            self._mcmc.likelihood = self._likelihood
            self._mcmc.current_state = self._likelihood.initial_state
            self._mcmc._throw_matrix = kwargs_dict['throw_matrix']
            self._mcmc.set_step_sizes(kwargs_dict['step_sizes'])

        
        # Jams only runs with MVG
        elif mcmc_type == "jams_mcmc_burnin" and likelihood_type == "multivariate_gaussian":
            self._mcmc = jams_mcmc_burner.jams_burner()
            self._mcmc.likelihood = self._likelihood
            self._mcmc.n_modes = self._likelihood.get_n_modes()
            
            self._mcmc.jump_epsilon = kwargs_dict['jump_epsilon']
            self._mcmc.alphas = kwargs_dict['alpha_arr']
            self._mcmc.beta = kwargs_dict['beta']
            self._mcmc.nsteps_burn = kwargs_dict['burn_in']
            self._mcmc.step_sizes = kwargs_dict['step_sizes']
            
        elif mcmc_type == "jams_mcmc" and likelihood_type == "multivariate_gaussian":
            self._mcmc = jams_mcmc.jams_mcmc()
            self._mcmc.likelihood = self._likelihood

            self._mcmc.n_modes = self._likelihood.get_n_modes()

            self._mcmc.jump_epsilon = kwargs_dict['jump_epsilon']
            self._mcmc.alphas = kwargs_dict['alpha_arr']
            self._mcmc.beta = kwargs_dict['beta']
            self._mcmc.current_state = self._likelihood.initial_state
            self._mcmc.step_sizes = kwargs_dict['step_sizes']


        elif mcmc_type == "hamiltonian_mcmc":
            self._mcmc = hamiltonian_mcmc.hamiltonian_mcmc()
            self._mcmc.likelihood = self._likelihood
            self._mcmc.time_step = kwargs_dict['time_step']
            self._mcmc.epsilon = kwargs_dict['epsilon']
            self._mcmc.mass_matrix = kwargs_dict['mass_matrix']
            self._mcmc.current_state = self._likelihood.initial_state
            
        else:
            raise ValueError("MCMC type not recognized, must be metropolis_hastings, jams_mcmc_burnin, jams_mcmc, or hamiltonian_mcmc.")

    @property
    def mcmc(self):
        return self._mcmc

    def __call__(self, nsteps: int) -> None:
        self._mcmc(nsteps)


    @property
    def mcmc_type(self) -> str:
        return self._mcmc_type
    
    def plot_mcmc(self, output_name: str, burnin, overlay=None) -> None:
        mcmc_chain = self._mcmc.step_array
        print(f"Plotting chain to file: {output_name}")
        n_params = mcmc_chain.shape[1]
        with PdfPages(f"{output_name}") as pdf:
            for param in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10,10))
                ax.hist(mcmc_chain[burnin:,param], bins=100)
                if overlay is not None:
                    ax.plot(overlay)
                ax.set_ylabel(f"Parameter {param}")
                ax.set_xlabel("Step")
                pdf.savefig(fig)
                plt.close(fig)
    