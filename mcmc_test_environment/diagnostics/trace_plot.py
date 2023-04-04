import numpy as np
from ..mcmc import mcmc_interface
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

'''
Class for getting trace plots
'''

class trace_plot:
    def __init__(self, imcmc: mcmc_interface) -> None:
        self._markov_chain = imcmc.mcmc.step_array
    
    def plot_traces(self, output_file_name: str) -> None:
        """
        Plot the trace plots
        """
        print("Plotting traces")
        n_params = self._markov_chain.shape[1]
        print(n_params)
        with PdfPages(f"{output_file_name}") as pdf:
            for param in tqdm(range(n_params)):
                fig, ax = plt.subplots(figsize=(10,10))
                ax.plot([self._markov_chain[step][param] for step in range(self._markov_chain.shape[0])])
                ax.set_ylabel(f"Parameter {param}")
                ax.set_xlabel("Step")
                pdf.savefig(fig)
                plt.close(fig)