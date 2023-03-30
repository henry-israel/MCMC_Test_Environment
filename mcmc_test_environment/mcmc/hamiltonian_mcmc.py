from mcmc import mcmc_base
import numpy as np
from typing import Type

'''
Hamiltonian MCMC base class, can run standard HMCMC
'''

class hamiltonian_mcmc(mcmc_base):
    def __init__(self) -> None:
        super().__init__()
        self._name = "hamiltonian mcmc"
        self._current_momentum = np.ndarray([])
        self._proposed_momentum = np.ndarray([])
        self._mass_matrix = np.ndarray([])
        self._epsilon = 0.1
        self._time_step = 3

        self._current_hamiltonian = 9999
        self._proposed_hamiltonian = 9999

    @property
    def epsilon(self) -> float:
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon

    @property
    def time_step(self) -> int:
        return self._time_step
    
    @time_step.setter
    def time_step(self, time_step: int) -> None:
        self._time_step = time_step

    @property
    def mass_matrix(self) -> np.ndarray:
        return self._mass_matrix
    
    @mass_matrix.setter
    def mass_matrix(self, mass_matrix: np.ndarray) -> None:
        if (mass_matrix.shape[0] != self._space_dim):
            raise ValueError("Mass matrix must be square and of same dimension as state space")
        else:
            self._mass_matrix = mass_matrix
        self._mass_matrix = mass_matrix

    def __call__(self, n_steps) -> None:
        if(self._mass_matrix.size == 0):
            self._mass_matrix = np.identity(self._space_dim)
        self._current_momentum = np.random.multivariate_normal(np.zeros(self._space_dim), self._mass_matrix)
        self._proposed_momentum = self._current_momentum
        return super().__call__(n_steps)
    
    def do_momentum_step(self) -> None:
        gradient = self._likelihood_space.get_gradient(self._proposed_state)
        self._proposed_momentum = self._proposed_momentum - 0.5*self._epsilon*gradient

    def do_position_step(self) -> None:
        self._proposed_state = self._proposed_state + self._epsilon*np.dot(self._mass_matrix, self._proposed_momentum)

    def propose_step(self) -> None:
        for _ in range(self._time_step):
            self.do_momentum_step()
            self.do_position_step()
            self.do_momentum_step()

    def accept_step(self) -> bool:
        self._current_hamiltonian = np.dot(self._current_momentum, self._current_momentum) + self._proposed_likelihood

        self._proposed_likelihood = self._likelihood_space.likelihood_function(self._proposed_state)
        self._proposed_hamiltonian = np.dot(self._current_momentum, self._current_momentum) + self._proposed_likelihood

        return min(1, np.exp(self._current_hamiltonian-self._proposed_hamiltonian))<np.random.uniform(0,1)
