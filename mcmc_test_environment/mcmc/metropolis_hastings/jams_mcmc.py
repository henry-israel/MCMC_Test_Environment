'''
Metroplis based JAMS MCMC algorithm
'''

from adaptive_metropolis_hastings import metropolis_hastings
from mcmc_base_class import mcmc_base
import numpy as np

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
        self._current_throw_matrix = np.ndarray([])

        self._is_jump = False
        self._alphas = np.ndarray([])

        self._total_steps_mode = np.ndarray([])

    def __call__(self, n_steps: int) -> None:
        super().__call__(n_steps)
    
    # Not excited for this but here are our setter functions
    @property
    def n_modes(self) -> int:
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes: int) -> None:
        self._n_modes = n_modes
        self._current_means = np.zeros((n_modes, self._space_dim))
        self._proposed_means = np.zeros((n_modes, self._space_dim))
        self._nominal_throw_matrix_arr = np.zeros((n_modes, self._space_dim, self._space_dim))+np.diag(np.ones(self._space_dim)*0.00001)
        self._alphas = np.ndarray([n_modes])
        self._total_steps_mode = np.zeros(n_modes)

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
    def current_covs(self) -> np.ndarray:
        return self._current_covs
    
    @current_covs.setter
    def current_covs(self, current_covs: np.ndarray) -> None:
        self._current_covs = current_covs

    def propose_step(self) -> None:
        if np.random.uniform(0,1)<self._jump_epsilon:
            self.do_jump_step()
        else:
            self.do_local_step()

    def do_jump_step(self) -> None:
        # Let's go and do a jump step
        self._isjump = True
        self._proposed_mode = np.random.randint([i for i in range(self._n_modes) if i != self._current_mode])
        chol_current = self._likelihood_space.indiv_likelihood[self._current_mode].get_covariance_root()
        chol_proposed = self._likelihood_space.indiv_likelihood[self._proposed_mode].get_covariance_root()

        mean_current = self._likelihood_space.indiv_likelihood[self._current_mode].mu
        mean_proposed = self._likelihood_space.indiv_likelihood[self._proposed_mode].mu

        self._proposed_state = mean_proposed + np.dot(np.dot(chol_proposed, np.linalg.inv(chol_current)), (self._current_state-mean_current))
 
    def do_local_step(self) -> None:
        self._is_jump = False
        self._proposed_state = self._current_state + np.random.multivariate_normal(np.zeros(self._space_dim), self._current_covs[self._current_mode])

    def update_local(self) -> None:
        self._current_throw_matrix *= np.exp((self._total_steps_mode[self._current_mode]/self._total_steps - 
                                              (1/self._n_modes-1))
                                              *self._total_steps_mode[self._current_mode]**(1/(self._n_modes-1)))
        self._current_throw_matrix = self._current_throw_matrix + np.diag(np.ones(self._space_dim)*0.0000001)

    def update_throw_matrix(self, index: int) -> None:
        
        self._current_means[self._current_mode] = (self._previous_means[self._current_mode]*(self._total_steps_mode[self._current_mode]-1) +
                                                    self._current_state)/self._total_steps_mode[self._current_mode]
        prev_mean_contribtion = np.outer(self._previous_means[self._current_mode], self._previous_means[self._current_mode])

        # Now we update our covariance
        # Numpy doesn't like +=, *=, -= or /= sadly
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] * (self._space_dim)/(2.68**2)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] + prev_mean_contribtion
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode]*(self._total_steps_mode[self._current_mode]-1)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode]+ np.outer(self._current_state, self._current_state)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode]/self._total_steps
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] +  np.outer(self._curr_mean, self._curr_mean)
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode]*(2.68**2)/self._space_dim
        self._nominal_throw_matrix_arr[self._current_mode] = self._nominal_throw_matrix_arr[self._current_mode] + np.diag(np.ones(self._space_dim)*self._beta)
        

