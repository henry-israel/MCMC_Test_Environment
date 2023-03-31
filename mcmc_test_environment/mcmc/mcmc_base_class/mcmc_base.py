'''
Base class for MCMC, contains common methods for all MCMC algorithms
'''

from typing import Any, List
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from likelihood import likelihood_base

class mcmc_base(ABC):
    """
    Base class implementation, bare bones structure with necessary interfacing
    """
    def __init__(self) -> None:
        self._name="MCMC"
        self._current_state = np.ndarray([])
        self._proposed_state = np.ndarray([])
        self._total_accepted_steps = 0
        self._space_dim = 0

        self._step_arr = np.ndarray([])
        self._likelihood_space = likelihood_base()

        self._current_likelihood = 9999999
        self._proposed_likelihood = 9999999
        self._total_steps = 0

    @property
    def step_count(self) -> int:
        return self._total_steps
    
    @step_count.setter
    def step_count(self, n_steps: int) -> None:
        self._total_steps = n_steps

    @property
    def current_state(self) -> Any:
        return self._current_state
    
    @current_state.setter
    def current_state(self, state: np.ndarray) -> None:
        self._current_state = state
        self._space_dim = len(state)


    def __call__(self, n_steps: int) -> None:
        print(f"Running MCMC for {n_steps} steps")
        print(f"Using {self._name} algorithm")

        self._step_arr = np.zeros((self._space_dim, n_steps))

        for step in tqdm(range(n_steps)):
            self._total_steps += 1
            self.propose_step()
            if self.accept_step():
                self._total_accepted_steps += 1
                self._current_state = self._proposed_state

            self._step_arr[step] = self._current_state
        print(f"Accepted {self._total_accepted_steps} steps out of {n_steps}")

    def set_likelihood_space(self, likelihood: likelihood_base) -> None:
        self._likelihood_space = likelihood
    
    def __str__(self) -> str:
        return f"Using {self._name} with {self._space_dim} dimensions"

    """
    Whilst these methods are different in each MCMC implementation,
    it is useful to have the scaffolding here
    """

    @abstractmethod
    def propose_step(self):
        pass

    @abstractmethod
    def accept_step(self) -> bool:
        pass
    
    """
    Let's add a nice iterable version so we can run this step by step
    """
    def __iter__(self):
        self._step_arr = np.zeros(self._space_dim)
        return self
    
    def __next__(self):
        self.propose_step()
        if self.accept_step():
            self._total_accepted_steps += 1
            self._current_state = self._proposed_state
        self._step_arr = np.append(self._step_arr, self._current_state)