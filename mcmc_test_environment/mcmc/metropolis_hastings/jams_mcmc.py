'''
Metroplis based JAMS MCMC algorithm
'''

from .metropolis_hastings import metropolis_hastings
import numpy as np
from typing import Callable
from random import choice

class jams_mcmc(metropolis_hastings):
    def __init__(self) -> None:
        super().__init__()
        self._name = "JAMS"
        self._n_modes = 1
        self._current_mode = 0
        self._proposed_mode = 0
        self._jump_epsilon = 0.1

        self._current_means = np.ndarray([])
        self._previous_means = np.ndarray([])

        self._nominal_throw_matrix_arr = np.ndarray([])
        self._throw_matrix = np.ndarray([])
        self._local_prior_dict = {
            0 : (lambda x : 1 for i in range(self._n_modes)),
            1 : (lambda x : 1 for i in range(self._n_modes))
        }

        self._is_jump = False
        self._alphas = np.ndarray([])
        self._beta = 0.1

        self._total_steps_mode = np.ndarray([])

        self._local_update_limiter = 200000
        self._global_update_limiter = 100

        # Value of likelihood from jump kernel
        self._current_jump_likelihood = 99999
        self._proposed_jump_likelihood = 99999
        self._total_steps = 1

    def __call__(self, n_steps: int) -> None:
        self._throw_matrix = np.diag(np.ones(self._space_dim)*self._beta)
        super().__call__(n_steps)
        print(f"Total steps/mode : {self._total_steps_mode}")
    
    # Not excited for this but here are our setter functions
    @property
    def n_modes(self) -> int:
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes: int) -> None:
        self._n_modes = n_modes
        self._current_means = np.zeros((n_modes, self._space_dim))
        self._previous_means = np.zeros((n_modes, self._space_dim))
        self._nominal_throw_matrix_arr = np.array([np.diag(np.ones(self._space_dim)*self._beta) for _ in range(n_modes)])
        self._alphas = np.ndarray([n_modes])
        self._total_steps_mode = np.zeros(n_modes)
        self._local_prior_dict = {i : (lambda x : 1) for i in range(self._n_modes)}

    @property
    def current_mode(self) -> int:
        return self._current_mode
    
    @current_mode.setter
    def current_mode(self, current_mode) -> None:
        self._current_mode = current_mode

    @property
    def jump_epsilon(self) -> float:
        return self._jump_epsilon
    
    @jump_epsilon.setter
    def jump_epsilon(self, jump_epsilon: float) -> None:
        self._jump_epsilon = jump_epsilon

    @property
    def current_means(self) -> np.ndarray:
        return self._current_means
    
    @current_means.setter
    def current_means(self, current_means: np.ndarray) -> None:
        self._current_means = current_means
    
    @property
    def alphas(self) -> np.ndarray:
        return self._alphas
    
    @alphas.setter
    def alphas(self, alphas: np.ndarray) -> None:
        self._alphas = alphas

    @property
    def beta(self) -> float:
        return self._beta
    
    @beta.setter
    def beta(self, beta: float) -> None:
        self._beta = beta

    @property
    def current_covs(self) -> np.ndarray:
        return self._nominal_throw_matrix_arr
    
    @current_covs.setter
    def current_covs(self, current_covs: np.ndarray) -> None:
        self._nominal_throw_matrix_arr = current_covs

    @property
    def local_update_limiter(self) -> int:
        return self._local_update_limiter
    
    @local_update_limiter.setter
    def local_update_limiter(self, local_update_limiter: int) -> None:
        self._local_update_limiter = local_update_limiter

    @property
    def global_update_limiter(self) -> int:
        return self._global_update_limiter
    
    @global_update_limiter.setter
    def global_update_limiter(self, global_update_limiter: int) -> None:
        self._global_update_limiter = global_update_limiter

    def set_cov_for_mode(self, cov: np.ndarray, mode: int) -> None:
        self._nominal_throw_matrix_arr[mode] = cov

    def set_local_prior_mode(self, prior: Callable, mode: int) -> None:
        new_prior = self._local_prior_dict

    def get_local_prior_mode(self, mode: int) -> Callable:
        return self._local_prior_dict[mode]

    def propose_step(self) -> None:
        if not self._is_jump and self._total_steps%self._global_update_limiter==0 and self._total_steps_mode[self._current_mode]>0:
            self.update_throw_matrix()
        if self._total_steps_mode[self._current_mode]<self._local_update_limiter:
            self.update_local()

        if np.random.uniform(0,1)<self._jump_epsilon:            
            self.do_jump_step()
        else:
            self.do_local_step()

    def do_jump_step(self) -> None:
        # Let's go and do a jump step
        self._is_jump = True
        self._proposed_mode = choice([i for i in range(self._n_modes) if i != self._current_mode])

        c_indiv_curr=self._likelihood_space.indiv_likelihood[self._current_mode]
        c_indiv_prop =self._likelihood_space.indiv_likelihood[self._proposed_mode]
        chol_current = c_indiv_curr.get_covariance_root()
        chol_proposed = c_indiv_prop.get_covariance_root()

        mean_current = c_indiv_curr.mu
        mean_proposed = c_indiv_prop.mu

        self._proposed_state = mean_proposed + np.dot(np.dot(chol_proposed, np.linalg.inv(chol_current)), (self._current_state-mean_current))
 
    def do_local_step(self) -> None:
        self._is_jump = False
        self._proposed_state = self._current_state + np.random.multivariate_normal(np.zeros(self._space_dim), self._throw_matrix)
        self._proposed_likelihood = self._likelihood_space.likelihood_function(self._proposed_state)

    def update_local(self) -> None:
        divisor = max(1, self._n_modes-1)

        self._throw_matrix *= np.exp((self._total_steps_mode[self._current_mode]/self._total_steps - 
                                              (1/divisor)))
        
        self._throw_matrix = self._throw_matrix + np.diag(np.ones(self._space_dim)*0.0000001)

    def update_throw_matrix(self) -> None:
        '''
        Updates total covariance for current mode
        '''
        self._current_means[self._current_mode] = (self._previous_means[self._current_mode]*(self._total_steps_mode[self._current_mode]-1) +
                                                    self._current_state)/self._total_steps_mode[self._current_mode]
        prev_mean_contribtion = np.outer(self._previous_means[self._current_mode], self._previous_means[self._current_mode])
        
        # Now we update our covariance
        # Numpy doesn't like +=, *=, -= or /= sadly
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] * (np.sqrt(self._space_dim))/(2.68**2)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] + prev_mean_contribtion
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] * (self._total_steps_mode[self._current_mode]-1)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] + np.outer(self._current_state, self._current_state)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] / self._total_steps_mode[self._current_mode]
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] + np.outer(self._current_means[self._current_mode], self._current_means[self._current_mode])
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] * (2.68**2)/np.sqrt(self._space_dim)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode]


        self._throw_matrix = self._nominal_throw_matrix_arr[self._current_mode]+np.diag(np.ones(self._space_dim)*self._beta)

    def accept_step(self) -> bool:
        self._total_steps_mode[self._current_mode] += 1
        if self._is_jump:
            if self.jump_acceptance():
                self._current_mode = self._proposed_mode
                self._current_state = self._proposed_state
                self._current_jump_likelihood = self._proposed_jump_likelihood
                self._throw_matrix = self._nominal_throw_matrix_arr[self._current_mode]
                return True
            return False
        else:
            if(super().accept_step()):
                return True
        
        return False


    def jump_acceptance(self) -> bool:
        matrix_prop = self._likelihood_space.indiv_likelihood[self._proposed_mode].covariance
        matrix_prop_determinant = np.linalg.det(matrix_prop)

        prior_prop = np.exp(self._local_prior_dict[self._proposed_mode](self._proposed_state))

        self._proposed_jump_likelihood = np.sqrt(matrix_prop_determinant)*prior_prop/(self._n_modes-1)
        return min(1, self._proposed_jump_likelihood/self._current_jump_likelihood)>np.random.uniform(0,1)
    
