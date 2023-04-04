import numpy as np
from ..mcmc import mcmc_interface
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

'''
Class for calculating self._autocorrelations
'''

class autocorrelation:
    def __init__(self, imcmc: mcmc_interface) -> None:
        self._markov_chain = imcmc.mcmc.step_array
        self._lag = 1000
        self._burnin =0
        self._nparams = self._markov_chain.shape[1]
        print(f"TOTAL PARAMS : {self._nparams}")
        self._autocorrelation = np.zeros((self._nparams, self._lag), dtype=float)


    @property
    def lag(self) -> int:
        return self._lag
    
    @lag.setter
    def lag(self, lag: int) -> None:
        self._lag = lag
        self._autocorrelation = np.zeros((self._nparams, self._lag), dtype=float)


    @property
    def burnin(self) -> int:
        return self._burnin
    
    @burnin.setter
    def burnin(self, burnin: int) -> None:
        self._burnin = burnin

    def __call__(self, lag=1000) -> np.ndarray:
        """
        Get the self._autocorrelation for each parameter
        """
        self._lag=lag
        self._autocorrelation = np.zeros((self._nparams, self._lag), dtype=float)

        for param in range(self._nparams):
            print(f"Making autocorrelation for parameter {param}")
            self._autocorrelation[param] = self._calc_autocorrelation(self._markov_chain[:,param])

        return self._autocorrelation

    def _calc_autocorrelation(self, param: np.ndarray) -> np.ndarray:
        """
        Calculate the self._autocorrelation for a single parameter
        """
        autocorrelation = np.zeros(self._lag)

        param_sums = np.mean(param[self._burnin:])
        n_steps = len(param)

        for lag in tqdm(range(self._lag)):
            denom = 0
            numerator =0
            for step in range(self._burnin, n_steps - lag):
                lag_term = param[lag+step] - param_sums
                diff = param[step]-param_sums
                prod = diff*lag_term

                numerator += prod
                denom     += diff*diff
            
            autocorrelation[lag] = numerator/denom

        return autocorrelation
    
    def plot_autocorrelation(self, output_file: str):
        """
        Plot the autocorrelation
        """
        print("Making autocorrelation plot")
        plt.figure()
        with PdfPages(f"{output_file}") as pdf:
            for param in tqdm(range(self._nparams)):
                fig, ax = plt.subplots(figsize=(10,10))
                ax.plot(self._autocorrelation[param])
                ax.set_ylabel("Autocorrelation")
                ax.set_xlabel(f"Lag {param}")
                pdf.savefig(fig)
                plt.close(fig)