"""
Base class for likelihood functionality
"""
import numpy as np
from typing import Callable

class likelihood_base():
    def __init__(self) -> None:
        self._likelihood_function = lambda x: np.exp(-np.sum(x**2)/2) # Bt default, use a gaussian likelihood
        self._hessian = np.ndarray([])

    @property
    def likelihood_function(self) -> Callable:
        return self._likelihood_function
    
    @likelihood_function.setter
    def likelihood_function(self, likelihood_function: Callable) -> None:
        """
        Set likelihood function as callable
        """
        self._likelihood_function = likelihood_function

    def get_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Gets multidimensional gradient of likelhood evaluated at a point (state)
        """
        return np.array([self._likelihood_function(state+np.array([1e-6,0]))-self._likelihood_function(state-np.array([1e-6,0]))])/2e-6

    def calculate_hessian(self, state: np.ndarray) -> None:
        """
        Gets multidimensional hessian of likelihood evaluated at a point (state)
        """
        llh_up = self._likelihood_function(state+np.array([1e-6,0]))
        llh_down = self._likelihood_function(state-np.array([1e-6,0]))
        llh_left = self._likelihood_function(state+np.array([0,1e-6]))
        llh_right = self._likelihood_function(state-np.array([0,1e-6]))

        self._hessian = np.array([[llh_up+llh_down-2*self._likelihood_function(state),
                                    llh_left+llh_right-2*self._likelihood_function(state)],
                                  [llh_left+llh_right-2*self._likelihood_function(state),
                                    llh_up+llh_down-2*self._likelihood_function(state)]])/1e-12

    def get_hessian(self, state: np.ndarray) -> np.ndarray:
        """
        Gets multidimensional hessian of likelihood evaluated at a point (state)
        """
        if(self._hessian is None):
            self.calculate_hessian(state)
        return self._hessian