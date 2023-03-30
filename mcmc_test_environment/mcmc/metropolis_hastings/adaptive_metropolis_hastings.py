from mcmc import metropolis_hastings
import numpy as np

class adaptive_metropolis_hastings(metropolis_hastings):
    def __init__(self) -> None:
        super().__init__()
        self._name = "adaptive metropolis hastings"
        self._prev_mean = np.ndarray([])
        self._curr_mean = np.ndarray([])
        self._beta = 0.00001
        self._throw_matrix = np.ndarray([])

    def __call__(self, n_steps: int) -> None:
        self._throw_matrix = np.diag(np.ones(self._space_dim)*self._beta)
        super().__call__(n_steps)

    def update_throw_matrix(self) -> None:
        """
        Let's update our throw matrix!
        """
        self._curr_mean = (self._prev_mean*(self._total_steps-1) + self._current_state)/self._total_steps
        prev_mean_contribtion = np.outer(self._prev_mean, self._prev_mean)

        # Now we update our covariance
        # Numpy doesn't like +=, *=, -= or /= sadly
        self._throw_matrix = self._throw_matrix * (self._space_dim)/(2.68**2)
        self._throw_matrix = self._throw_matrix + prev_mean_contribtion
        self._throw_matrix = self._throw_matrix*(self._total_steps-1)
        self._throw_matrix = self._throw_matrix+ np.outer(self._current_state, self._current_state)
        self._throw_matrix = self._throw_matrix/self._total_steps
        self._throw_matrix = self._throw_matrix +  np.outer(self._curr_mean, self._curr_mean)
        self._throw_matrix = self._throw_matrix*(2.68**2)/self._space_dim
        self._throw_matrix = self._throw_matrix + np.diag(np.ones(self._space_dim)*self._beta)

    def propose_step(self) -> None:
        """
        Same proposal as metropolis hastings, but we update our throw matrix
        """
        self.update_throw_matrix()
        super().propose_step()


