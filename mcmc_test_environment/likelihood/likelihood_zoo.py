"""
Collection of likelihoods for use in MCMC-TE.
"""

from likelihood import likelihood_base
import numpy as np
from typing import Callable, TypeVar

class gaussian_likelihood(likelihood_base):
    def __init__(self) -> None:
        super().__init__()
        self._covariance = np.ones((2,2))
        self._mu = np.zeros(2)

    def set_likelihood_function(self, mu: np.float64, covariance: np.ndarray) -> None:
        self._mu = mu
        self._covariance = covariance
        self._likelihood_function = lambda x: np.exp(-np.sum((x-self._mu)**2)/2)/np.sqrt(np.linalg.det(2*np.pi*self._covariance))

class multi_modal_gaussian(likelihood_base):
    def __init__(self) -> None:
        super().__init__()
        self._covariance = np.ones((2,2,2))
        self._mu = np.zeros((2,2))
        self._individual_likelihoods = np.empty(2, dtype=gaussian_likelihood)

    def set_likelhood_function(self, mu: np.ndarray, covariance: np.ndarray) -> None:
        self._mu = mu
        self._covariance = covariance
        for i in range(len(mu)):
            self._individual_likelihoods[i] = gaussian_likelihood()
            self._individual_likelihoods[i].set_likelihood_function(self._mu[i], self._covariance[i])
        self._likelihood_function = lambda x: np.sum([self._individual_likelihoods[i].likelihood_function(x) for i in range(len(mu))])

    def get_indiv_likelihood(self, index):
        return self._individual_likelihoods[index]
    
class poisson_likelihood(likelihood_base):
    def __init__(self) -> None:
        super().__init__()
        self._lambda = np.ones(2)

    def set_likelihood_function(self, lam: np.ndarray) -> None:
        self._lambda = lam
        self._likelihood_function = lambda x: np.exp(-np.sum(self._lambda)+np.sum(x*np.log(self._lambda)))


class likelihood_interface:
    def __init__(self, likelihood_name: str) -> None:
        if likelihood_name == "gaussian":
            self._likelihood = gaussian_likelihood()
        elif likelihood_name == "multi_modal":
            self._likelihood = multi_modal_gaussian()
        elif likelihood_name == "poisson":
            self._likelihood = poisson_likelihood()
        else:
            raise ValueError("Likelihood not recognized.")
        
    def get_likelihood(self) -> likelihood_base:
        return self._likelihood # type: ignore
    