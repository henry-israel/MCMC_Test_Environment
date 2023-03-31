"""
Base class for likelihood functionality
"""
import numpy as np
from scipy.optimize import minimize
from typing import Callable

class likelihood_base():
    def __init__(self) -> None:
        self._likelihood_function = lambda x: np.exp(-np.sum(x**2)/2) # Bt default, use a gaussian likelihood
        self._covariance = np.ndarray([])
        self._prior = lambda x: 1.0 # Default prior is uniform
        self._initial_state = np.zeros(2)

    def get_n_modes(self):
        return 1
    
    @property
    def initial_state(self) -> np.ndarray:
        return self._initial_state

    @property
    def likelihood_function(self) -> Callable:
        return self._likelihood_function
    
    @likelihood_function.setter
    def likelihood_function(self, likelihood_function: Callable) -> None:
        """
        Set likelihood function as callable
        """
        self._likelihood_function = likelihood_function

    @property
    def prior(self) -> Callable:
        return self._prior
    
    @prior.setter
    def prior(self, prior: Callable) -> None:
        """
        Set prior probabilities for each dimension
        """
        self._prior = prior

    def get_gradient_variable(self, state: np.ndarray, index: int, state_llh: float) -> float:
        state_plus = state.copy()
        state_plus[index] += 1e-6
        return (self._likelihood_function(state_plus)-state_llh)/1e-6

    def get_full_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Gets multidimensional gradient of likelhood evaluated at a point (state)
        """
        grad_vec = np.zeros(len(state))
        state_llh = self._likelihood_function(state)

        for i in range(len(state)):
            self.get_gradient_variable(state, i, state_llh)
        return grad_vec

    def calculate_local_covariance(self, initial_state: np.ndarray) -> None:
        """
        Gets multidimensional hessian of likelihood evaluated at a point (state)
        """
        result = minimize(lambda p: -self._likelihood_function(p), initial_state, method='BFGS')
        self._covariance = result.hess_inv

    def get_local_covariance(self, state: np.ndarray) -> np.ndarray:
        """
        Gets multidimensional hessian of likelihood evaluated at a point (state)
        """
        if(self._covariance is None):
            self.calculate_local_covariance(state)
        return self._covariance
    
    def get_covariance_root(self):
        return np.linalg.cholesky(self._covariance)