"""
Collection of likelihoods for use in MCMC-TE.
"""

from .likelihood_base import likelihood_base
import numpy as np
from typing import Callable, TypeVar
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

class gaussian_likelihood(likelihood_base):
    def __init__(self) -> None:
        super().__init__()
        self._covariance = np.ones((2,2))
        self._mu = np.zeros(2)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mu(self) -> np.ndarray:
        return self._mu # type: ignore

    def set_likelihood_function(self, mu: np.ndarray, covariance: np.ndarray) -> None:
        self._mu = mu
        self._initial_state = mu
        self._covariance = covariance
        self._likelihood_function = multivariate_normal(mean=self._mu, cov=self._covariance).logpdf

    def __str__(self) -> str:
        return f"Likelihood function gaussian class with mean {self._mu} and covariance {self._covariance}"

class multi_modal_gaussian(likelihood_base):
    def __init__(self) -> None:
        super().__init__()
        self._covariance = np.ones((2,2,2))
        self._mu = np.zeros((2,2))
        self._individual_likelihoods = np.empty(2, dtype=gaussian_likelihood)

    def get_n_modes(self):
        return len(self._mu)

    def set_likelihood_function(self, mu: np.ndarray, covariance: np.ndarray) -> None:
        self._mu = mu
        print(self._mu)

        self._initial_state = mu[0]
        self._space_dim = len(mu[0])

        self._covariance = covariance

        for i in range(len(mu)):
            self._individual_likelihoods[i] = gaussian_likelihood()
            self._individual_likelihoods[i].set_likelihood_function(self._mu[i], self._covariance[i])
            self._individual_likelihoods[i].priors = self._prior

        self._likelihood_function = lambda x: logsumexp([self._individual_likelihoods[i].likelihood_function(x) for i in range(len(mu))])

    @property
    def indiv_likelihood(self) -> np.ndarray:
        return self._individual_likelihoods
    
    def __str__(self) -> str:
        return f"Likelihood function multi-variate gaussian class with mean {self._mu} and covariance {self._covariance}"


class likelihood_interface:
    def __init__(self, likelihood_name: str) -> None:
        if likelihood_name == "gaussian":
            self._likelihood = gaussian_likelihood()
        elif likelihood_name == "multivariate_gaussian":
            self._likelihood = multi_modal_gaussian()
        else:
            raise ValueError("Likelihood not recognized.")

    def set_gaussian_prior(self, mu: np.ndarray, covariance: np.ndarray) -> None:
        self._likelihood.prior = lambda x: [np.exp(-np.sum((x-mu[i])**2)/2)/np.sqrt(np.linalg.det(2*np.pi*covariance[i])) for i in range(len(mu))]

    def get_likelihood(self) -> likelihood_base:
        return self._likelihood # type: ignore
    