import numpy as np
import emcee
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, freeze_support
from config import config
import torch
from ..utils import negative_log_likelihood

class SamplingMethod:
    def __init__(self) -> None:
        """
        Base class for sampling methods.
        """
        pass

class MCMCSamplingMethod(SamplingMethod):
    def __init__(self):
        """
        Initialize the MCMCSamplingMethod class.

        Parameters:
        num_samples (int): Number of samples to generate.
        burn_in_steps (int): Number of burn-in steps.
        num_steps_between_samples (int): Number of steps between each sample.
        nwalkers (int): Number of walkers in the MCMC ensemble.
        ndim (int): Number of dimensions in the parameter space.
        perturbation_noise (float): perturbation noise.
        """
        self.num_samples = config["inferenceParams"]["mcmc"]["numSamples"]
        self.burn_in_steps = config["inferenceParams"]["mcmc"]["burnInSteps"]
        self.num_steps_between_samples = config["inferenceParams"]["mcmc"]["numStepsBetweenSamples"]
        self.nwalkers = config["inferenceParams"]["mcmc"]["nWalkers"]
        self.perturbation_noise = config["inferenceParams"]["mcmc"]["perturbationNoise"]

    def sample(self, initial_state, Y_t, X_t, Z_t):
        """
        Generate samples using the MCMC method.

        Parameters:
        initial_state (np.ndarray): Initial state of the parameters.

        Returns:
        np.ndarray: Generated samples.
        """
        # Initial positions of the walkers
        ndim = len(initial_state)
        pos = initial_state + self.perturbation_noise * np.random.randn(self.nwalkers, ndim)
        
        with Pool(processes=cpu_count()) as pool:
            # Set up the MCMC sampler
            sampler = emcee.EnsembleSampler(self.nwalkers, ndim, negative_log_likelihood, args=(Y_t, X_t, Z_t), pool=pool)

            # Run the MCMC sampler
            sampler.run_mcmc(pos, self.num_samples, progress=True)

        # Get the samples
        samples = sampler.get_chain(discard=self.burn_in_steps, thin=self.num_steps_between_samples, flat=True)
        return samples

    def plot_traces(self, sampler):
        # """
        # Plot the traces of the MCMC samples.

        # Parameters:
        # sampler (emcee.EnsembleSampler): The MCMC sampler object.
        # """
        # fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        # for i in range(ndim):
        #     ax = axes[i]
        #     ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
        #     ax.set_xlim(0, len(sampler.get_chain()))
        #     ax.set_ylabel(f"Parameter {i}")
        # axes[-1].set_xlabel("Step number")
        # plt.show()
        pass

    def extract_parameters(self, initial_state, Y_t, X_t, Z_t):
        """
        Extract mean parameter estimates from the samples.

        Parameters:

        Returns:
        tuple: Extracted parameters (G, eta, zeta).
        """
        samples = self.sample(initial_state, Y_t, X_t, Z_t)
        mean_params = np.mean(samples, axis=0)

        print("mean_params", mean_params)
        G, *coeffs = mean_params
        eta = np.array(coeffs[:X_t.shape[1]])
        zeta = np.array(coeffs[X_t.shape[1]:])
        # Convert each element to torch tensor individually
        G_tensor = torch.tensor([G], requires_grad=False)
        eta_tensor = torch.tensor(eta, requires_grad=False)
        zeta_tensor = torch.tensor(zeta, requires_grad=False)

        # Return them as a list of tensors or concatenate them as needed
        return [G_tensor, eta_tensor, zeta_tensor]
    

    # Define the log probability function
    def log_probability(self, params, Y_t, X_t, Z_t):
        G, *coeffs = params
        eta = np.array(coeffs[:X_t.shape[1]])
        zeta = np.array(coeffs[X_t.shape[1]:])

        T = len(Y_t)
        theta_t = np.zeros(T)
        log_prob = 0

        for t in range(T):
            if t > 0:
                theta_t[t] = G * theta_t[t-1] + np.dot(Z_t[t-1], zeta / 2)
            predicted_Y_t = theta_t[t] + np.dot(X_t[t], eta) + np.dot(Z_t[t], zeta / 2)
            log_prob -= 0.5 * ((Y_t[t] - predicted_Y_t)**2)
        
        return log_prob
