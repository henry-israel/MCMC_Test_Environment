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

    def update_total_covariance(self):
        for i, chain in enumerate(self._jams_chains):
            self._local_covariance[i] = lambda x: np.random.multivariate_normal(x, 
                                                chain._current_means[i], chain._current_covs[i])/self._n_modes

        self._covariance_factor = lambda x: sum([self._local_covariance[i](x) for i in range(self._n_modes)])

    def __call__(self, nsteps) -> Any:

        for i, chain in enumerate(self._jams_chains):
            chain.likelihood = self._likelihood
            chain.n_modes = self._n_modes
            chain.current_mode = i
            self._prior_array[i] = chain.get_local_prior_mode(i)

        for _ in tqdm(range(nsteps)):
            for chain in self._jams_chains:
                chain.propose_step()
                if(chain.accept_step):
                    chain._total_accepted_steps += 1
                    chain._current_state = chain._proposed_state

            # Now we update our priors!
            self.update_total_covariance()
            for i, chain in enumerate(self._jams_chains):
                chain.set_local_prior_mode(self._prior_array[i], i)

        return self._prior_array