from .metropolis_hastings import metropolis_hastings
from random import choice
import numpy as np

class metropolis_with_explicit_jump(metropolis_hastings):
    def __init__(self) -> None:
        super().__init__()
        self._current_mode = 0
        self._proposed_mode = 0

    def __call__(self, n_steps: int) -> None:
        self._n_modes = self._likelihood_space.get_n_modes()
        return super().__call__(n_steps)
    
    def propose_step(self) -> None:
        self._proposed_mode = choice([i for i in range(self._n_modes)])
        if(self._proposed_mode!=self._current_mode):
            self.jump_to_new_mode()
        return super().propose_step()
    
    def accept_step(self) -> bool:
        if super().accept_step():
            self._current_mode = self._proposed_mode
            return True
        return False

    def jump_to_new_mode(self) -> None:
        c_indiv_curr=self._likelihood_space.indiv_likelihood[self._current_mode]
        c_indiv_prop =self._likelihood_space.indiv_likelihood[self._proposed_mode]
        chol_current = c_indiv_curr.get_covariance_root()
        chol_proposed = c_indiv_prop.get_covariance_root()
        mean_current = c_indiv_curr.mu
        mean_proposed = c_indiv_prop.mu

        self._proposed_state = mean_proposed + np.dot(np.dot(chol_proposed, np.linalg.inv(chol_current)), (self._current_state-mean_current))
 