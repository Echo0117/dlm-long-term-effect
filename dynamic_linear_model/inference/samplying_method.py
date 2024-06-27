import numpy as np
import emcee
import matplotlib.pyplot as plt

class SamplingMethod:
    def __init__(self) -> None:
        """
        Base class for sampling methods.
        """
        pass

class MCMCSamplingMethod(SamplingMethod):
    def __init__(self, model, num_samples, burn_in_steps, num_steps_between_samples, nwalkers, ndim, log_prob_fn, args):
        """
        Initialize the MCMCSamplingMethod class.

        Parameters:
        model (object): The model used for state transitions.
        num_samples (int): Number of samples to generate.
        burn_in_steps (int): Number of burn-in steps.
        num_steps_between_samples (int): Number of steps between each sample.
        nwalkers (int): Number of walkers in the MCMC ensemble.
        ndim (int): Number of dimensions in the parameter space.
        log_prob_fn (function): Log probability function for the MCMC sampler.
        args (tuple): Additional arguments for the log probability function.
        """
        self.model = model
        self.num_samples = num_samples
        self.burn_in_steps = burn_in_steps
        self.num_steps_between_samples = num_steps_between_samples
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.args = args

    def sample(self, initial_state):
        """
        Generate samples using the MCMC method.

        Parameters:
        initial_state (np.ndarray): Initial state of the parameters.

        Returns:
        np.ndarray: Generated samples.
        """
        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob_fn, args=self.args)

        # Initial positions of the walkers
        pos = initial_state + 1e-4 * np.random.randn(self.nwalkers, self.ndim)

        # Run the MCMC sampler
        sampler.run_mcmc(pos, self.num_samples, progress=True)

        # Get the samples
        samples = sampler.get_chain(discard=self.burn_in_steps, thin=self.num_steps_between_samples, flat=True)
        return samples

    def plot_traces(self, sampler):
        """
        Plot the traces of the MCMC samples.

        Parameters:
        sampler (emcee.EnsembleSampler): The MCMC sampler object.
        """
        fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(sampler.get_chain()))
            ax.set_ylabel(f"Parameter {i}")
        axes[-1].set_xlabel("Step number")
        plt.show()

    def extract_parameters(self, samples):
        """
        Extract mean parameter estimates from the samples.

        Parameters:
        samples (np.ndarray): The generated samples.

        Returns:
        tuple: Extracted parameters (G, sigma_v, sigma_w, eta, zeta).
        """
        mean_params = np.mean(samples, axis=0)
        G, log_sigma_v, log_sigma_w, *coeffs = mean_params
        sigma_v = np.exp(log_sigma_v)
        sigma_w = np.exp(log_sigma_w)
        eta = np.array(coeffs[:self.model.X_t.shape[1]])
        zeta = np.array(coeffs[self.model.X_t.shape[1]:])
        return G, sigma_v, sigma_w, eta, zeta
    
    def log_probability(theta, Y_t, X_t, Z_t):
        G, log_sigma_v, log_sigma_w, *coeffs = theta
        sigma_v = np.exp(log_sigma_v)
        sigma_w = np.exp(log_sigma_w)
        eta = np.array(coeffs[:X_t.shape[1]])
        zeta = np.array(coeffs[X_t.shape[1]:])
        
        if sigma_v <= 0 or sigma_w <= 0:
            return -np.inf
        
        T = len(Y_t)

        # Initial state
        theta_t = np.zeros(T)
        
        # Log-likelihood
        log_prob = 0

        # Iterate through each time step
        for t in range(T):
            if t > 0:
                # State transition equation
                theta_t[t] = G * theta_t[t-1] + np.dot(Z_t[t-1], zeta/2)
            
            # Observation equation
            predicted_Y_t = theta_t[t] + np.dot(X_t[t], eta) + np.dot(Z_t[t], zeta/2) 
            
            # Log-likelihood contribution
            log_prob += -0.5 * np.log(2 * np.pi * sigma_v**2) - 0.5 * ((Y_t[t] - predicted_Y_t)**2 / sigma_v**2)
        
        return log_prob
