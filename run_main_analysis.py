from mcmc_test_environment.mcmc import mcmc_interface
import numpy as np
from mcmc_test_environment.diagnostics.trace_plot import trace_plot
from mcmc_test_environment.diagnostics.autocorrelation import autocorrelation


if __name__=="__main__":
    print("Running main analysis")

    TOTAL_STEPS=1000000

    # Set up the mcmc objects
    SPACEDIM = 2
    mu = [np.zeros((SPACEDIM)), 10*np.ones((SPACEDIM))]
    covariance_matrix = [np.diag(np.ones(SPACEDIM)), 3*np.diag(np.ones(SPACEDIM))]
    step_sizes = np.ones(SPACEDIM)

    # # metropolis hastings objects
    metropolis_hastings_obj = mcmc_interface.mcmc_interface(mcmc_type="jump_metropolis", likelihood_type="multivariate_gaussian", step_sizes=step_sizes, mu=mu, covariance=covariance_matrix, throw_matrix=covariance_matrix[0])
    # adaptive_mcmc_obj = mcmc_interface.mcmc_interface(mcmc_type="adaptive_metropolis_hastings", likelihood_type="multivariate_gaussian", step_sizes=step_sizes, mu=mu, covariance=covariance_matrix)

    # # JAMs Objects
    alpha = np.ones(SPACEDIM)/SPACEDIM
    jams_obj = mcmc_interface.mcmc_interface(mcmc_type="jams_mcmc_burnin", likelihood_type="multivariate_gaussian",
                                              mu=mu, covariance=covariance_matrix, alpha_arr=alpha, beta=0.1,
                                                jump_epislon=0.5, step_sizes=step_sizes, burn_in=200000)

    # Hamiltonian MCMC
    hamiltonian_mcmc_obj = mcmc_interface.mcmc_interface("hamiltonian_mcmc", "multivariate_gaussian", mu=mu, 
                                                         covariance=covariance_matrix, time_step=40, epsilon=0.1, 
                                                         mass_matrix=np.identity(SPACEDIM)*0.268**2/np.sqrt(SPACEDIM))

    # adaptive_mcmc_obj(100000)
    # throw = adaptive_mcmc_obj.mcmc.throw_matrix


    # Run the mcmc objects
    mcmc_arr = [metropolis_hastings_obj, jams_obj]
    for mcmc_obj in mcmc_arr:
        mcmc_obj(TOTAL_STEPS)
        # Plot the object
        t = trace_plot(mcmc_obj)
        t.plot_traces(f"trace_{mcmc_obj.mcmc_type}.pdf")

        # a = autocorrelation(mcmc_obj)
        # a(1000)
        # a.plot_autocorrelation(f"autocorrelation_{mcmc_obj.mcmc_type}.pdf")

        mcmc_obj.plot_mcmc(f"{mcmc_obj.mcmc_type}.pdf", burnin=100000)