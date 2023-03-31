from mcmc import jams_mcmc
from likelihood import likelihood_base, likelihood_zoo
from typing import Any
import numpy as np
from tqdm import tqdm

'''
Class to handle several JAMS chains and run them in parallel
'''
class jams_burner():
    def __init__(self) -> None:
        self._likelihood = None 
        self._n_modes = 0
        self._jams_chains = np.ndarray([])
        self._prior_array = np.ndarray([])
        self._covariance_factor = None
        self._local_covariance = np.ndarray([])
        self._alpha_arr = np.ndarray([])
        self._beta = 0.0000001
        self._jump_epsilon = 0.1
        nsteps_burn = 10000

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
    
    @n_modes.setter
    def n_modes(self, n_modes: int) -> None:
        self._n_modes = n_modes
        self._jams_chains = [jams_mcmc() for _ in range(n_modes)]
        self._prior_array = np.ndarray([n_modes])
        self._local_covariance = np.ndarray([n_modes])

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


    def update_total_covariance(self):
        for i, chain in enumerate(self._jams_chains):
            self._local_covariance[i] = lambda x: np.random.multivariate_normal(x, 
                                                chain._current_means[i], chain._current_covs[i])/self._n_modes

        self._covariance_factor = lambda x: sum([self._local_covariance[i](x) for i in range(self._n_modes)])

    def initalise_chains(self):
        chain = jams_mcmc()
        chain.jump_epsilon = 0
        chain.likelihood = self.likelihood
        chain.alphas = self._alpha_arr
        chain.beta = self._beta
        chain.n_modes = self._n_modes

    def __call__(self, n_steps_run) -> Any:
        
        print(f"RUNNING JAMS WITH {self._nsteps_burn} of burnin")
        for i, chain in enumerate(self._jams_chains):
            chain.current_state = self._likelihood.indiv_likelihood[i].mu
            chain.likelihood = self._likelihood
            chain.n_modes = self._n_modes
            chain.current_mode = i
            self._prior_array[i] = chain.get_local_prior_mode(i)

        for _ in tqdm(range(self._nsteps_burn)):
            for chain in self._jams_chains:
                chain.propose_step()
                if(chain.accept_step):
                    chain._total_accepted_steps += 1
                    chain._current_state = chain._proposed_state

            # Now we update our priors!
            self.update_total_covariance()
            for i, chain in enumerate(self._jams_chains):
                chain.set_local_prior_mode(self._prior_array[i], i)

        # Okay now we've burnt in we can actually run a single chain
        print("Burnt in, now running JAMS chain")

        new_chain = jams_mcmc()
        new_chain.likelihood = self._likelihood
        new_chain.n_modes = self._n_modes
        
        new_chain.current_state = self._jams_chains[0]._current_state
        new_chain.jump_epsilon = self._jump_epsilon
        new_chain.beta = self._beta
        new_chain.alphas = self._alpha_arr

        for i in range(self._n_modes):
            new_chain.set_local_prior_mode(self._prior_array[i], i)
            new_chain.current_covs[i] = self._jams_chains[i].current_covs[i]
            new_chain.current_means[i] = self._jams_chains[i].current_means[i]
            new_chain.alphas[i] = self._jams_chains[i].alphas[i]

        new_chain(n_steps_run)

        # Returns the chain so we can mess with it!
        return new_chain
