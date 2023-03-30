from mcmc import mcmc_base 
import numpy as np


class metropolis_hastings(mcmc_base):
    def __init__(self) -> None:
        super().__init__()
        self._name = "metropolis hastings"
        self._step_sizes = np.ndarray([])

    def __call__(self, n_steps: int) -> None:
        super().__call__(n_steps)

    def set_step_sizes(self, step_sizes: np.ndarray = np.ndarray([])) -> None:
        if len(step_sizes)==0:
            self._step_sizes = np.ones(self._space_dim)
        elif len(step_sizes)!=self._space_dim:
            raise ValueError("ERROR::Step sizes must be of same dimension as likelihood space")
        else:
            self._step_sizes = step_sizes
            print(f"Set step sizes to \n {step_sizes}")

    def set_throw_matrix(self, throw_matrix: np.ndarray) -> None:
        if len(self._throw_matrix)!=self._space_dim:
            raise ValueError("ERROR::Throw matrix must be of same dimension as likelihood space")
        self._throw_matrix = throw_matrix
    

    def propose_step(self) -> None:
        self._proposed_state = self._current_state + np.random.multivariate_normal(np.zeros(self._space_dim), self._throw_matrix)*self._step_sizes
        self._proposed_likelihood = self._likelihood_space.likelihood_function(self._proposed_state)

    def accept_step(self) -> bool:  
        likelihood_ratio = self._proposed_likelihood-self._current_likelihood
        random_number = np.random.uniform(0,1)
        return min(1, np.exp(likelihood_ratio))<random_number    
    