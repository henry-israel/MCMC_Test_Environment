from .jams_mcmc import jams_mcmc
from ...likelihood import likelihood_base, likelihood_zoo
from typing import Any
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

'''
Class to handle several JAMS chains and run them in parallel
'''
class jams_burner():
    def __init__(self) -> None:
        self._likelihood = None 
        self._n_modes = 0
        self._jams_chains = np.ndarray([])
        self._prior_array = np.ndarray([])
        self._alpha_arr = np.ndarray([])
        self._beta = 0.0000001
        self._jump_epsilon = 0.1
        self._nsteps_burn = 10000
        self._step_arr = np.ndarray([])

        self._step_sizes = np.ndarray([])

        self._local_priors = lambda x, i: 1+x+i

    @property
    def nsteps_burn(self) -> int:
        return self._nsteps_burn
    
    @nsteps_burn.setter
    def nsteps_burn(self, nsteps_burn: int) -> None:
        self._nsteps_burn = nsteps_burn

    @property
    def likelihood(self) -> likelihood_base:
        return self._likelihood #type: ignore
    
    @likelihood.setter
    def likelihood(self, likelihood: likelihood_base) -> None:
        self._likelihood = likelihood #type: ignore
    
    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def step_sizes(self) -> np.ndarray:
        return self._step_sizes
    
    @step_sizes.setter
    def step_sizes(self, step_sizes: np.ndarray) -> None:
        self._step_sizes = step_sizes

    @n_modes.setter
    def n_modes(self, n_modes: int) -> None:
        self._n_modes = n_modes
        self._jams_chains = [jams_mcmc() for _ in range(n_modes)]
        self._prior_array = np.empty(n_modes)
        self._local_covariance = np.empty(n_modes)

    @property
    def jump_epsilon(self) -> float:
        return self._jump_epsilon
    
    @jump_epsilon.setter
    def jump_epsilon(self, jump_epsilon: float) -> None:
        self._jump_epsilon = jump_epsilon

    @property
    def alphas(self) -> np.ndarray:
        return self._alpha_arr
    
    @alphas.setter
    def alphas(self, alpha_arr: np.ndarray) -> None:
        self._alpha_arr = alpha_arr

    @property
    def beta(self) -> float:
        return self._beta
    
    @beta.setter
    def beta(self, beta: float) -> None:
        self._beta = beta

    @property
    def step_array(self) -> np.ndarray:
        return self._step_arr

    def update_total_covariance(self) -> None:
        mv_arr = [multivariate_normal(chain._current_means[i], chain._nominal_throw_matrix_arr[i]) for i, chain in enumerate(self._jams_chains)]
        cov_factor = lambda x,i : mv_arr[i].pdf(x)
        self._local_priors = lambda x,i : cov_factor(x,i) * self._local_priors(x,i)

    def initialise_chains(self, mode: int) -> jams_mcmc:
        chain = jams_mcmc()
        chain.current_mode = mode
        chain.jump_epsilon = 0
        chain.likelihood = self.likelihood
        chain.alphas = self._alpha_arr
        chain.beta = self._beta
        chain.global_update_limiter = 10
        chain.n_modes = self._n_modes
        chain.initialise_step_array(self._nsteps_burn)
        return chain

    def __call__(self, n_steps_run: int) -> Any:
        
        print(f"RUNNING JAMS WITH {self._nsteps_burn} steps of burnin")
        self._jams_chains = np.array([self.initialise_chains(i) for i in range(self._n_modes)])
        
        for i, chain in enumerate(self._jams_chains):
            self._local_priors = lambda x, i : chain.get_local_prior_mode(i)(x)
            chain.current_mode = i
            chain.current_state = self._likelihood.indiv_likelihood[i].initial_state
            print(f"Current state : {chain.current_state}")
            chain.set_local_prior_mode(self._local_priors, i)
            chain(self._nsteps_burn)

            self.plot_traces(chain, f"burnt_in_chain_{i}.pdf")

        print("Burnt in, now running JAMS chain")

        new_chain = jams_mcmc()
        new_chain.likelihood = self._likelihood
        new_chain.n_modes = self._n_modes
        new_chain.current_state = self._jams_chains[0]._current_state
        new_chain.jump_epsilon = self._jump_epsilon
        new_chain.beta = self._beta
        new_chain.alphas = self._alpha_arr
        new_chain.current_covs = np.empty(self._n_modes, dtype=object)
        new_chain.set_step_sizes(self._step_sizes)

        for i in range(self._n_modes):
            new_chain.set_local_prior_mode(self._prior_array[i], i)
            new_chain.current_covs[i] = self._jams_chains[i].current_covs[i]
            new_chain.current_means[i] = self._jams_chains[i].current_means[i]
            # new_chain.alphas[i] = self._jams_chains[i].alphas[i]

        new_chain(n_steps_run)

        # Returns the chain so we can mess with it!
        self._step_arr = new_chain.step_array
        return new_chain

    @classmethod
    def plot_traces(cls, mcmc_obj, output_file_name: str) -> None:
        """
        Plot the trace plots
        """
        print("Plotting traces")
        markov_chain = mcmc_obj.step_array
        print(markov_chain.shape)
        n_params = markov_chain.shape[1]
        print(n_params)
        with PdfPages(f"{output_file_name}") as pdf:
            for param in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10,10))
                ax.plot([markov_chain[step][param] for step in range(markov_chain.shape[0])])
                ax.set_ylabel(f"Parameter {param}")
                ax.set_xlabel("Step")
                pdf.savefig(fig)
                plt.close(fig)