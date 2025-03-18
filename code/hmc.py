import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from antikythera_model import AntikytheraModel
from optimize_antikythera import load_hole_data

# Load hole data
data_filename = "data/1-Fragment_C_Hole_Measurements.csv"
data, measured_positions, hole_indices = load_hole_data(data_filename)



def bayesian_model(covariance_model="isotropic"):
    # Ring parameters
    r = pyro.sample("r", dist.Normal(77.0, 5.0))  # Prior on ring radius
    N = pyro.sample("N", dist.Normal(354.0, 10.0))  # Prior on number of holes

    # Section parameters
    alpha_dict = {
        section_id: pyro.sample(f"alpha_{section_id}", dist.Normal(-145.0 - section_id, 5))
        for section_id in measured_positions.keys()
    }

    # Instantiate the model
    model = AntikytheraModel(section_ids=measured_positions.keys(), 
                           covariance_type=covariance_model)
    
    # Only sample the first hole from each section
    if covariance_model == "isotropic":
        # Sample the global sigma parameter once
        sigma = pyro.sample("sigma", dist.HalfNormal(5.0))  # Standard deviation
        
        for section_id in measured_positions.keys():
            if len(measured_positions[section_id]) > 0:  
                hole_idx = 0
                x_obs, y_obs = measured_positions[section_id][hole_idx]
                
                mean_x, mean_y = model.forward(section_id, hole_idx)  # Predicted location
                
                # Isotropic Gaussian likelihood
                cov_matrix = torch.eye(2, dtype=torch.float64) * sigma**2
                pyro.sample(
                    f"obs_{section_id}_{hole_idx}",
                    dist.MultivariateNormal(torch.tensor([mean_x, mean_y]), covariance_matrix=cov_matrix),
                    obs=torch.tensor([x_obs, y_obs])
                )
    else:  # anisotropic
        # Sample the global sigma parameters once
        sigma_r = pyro.sample("sigma_r", dist.HalfNormal(5.0))  # Radial std dev
        sigma_t = pyro.sample("sigma_t", dist.HalfNormal(5.0))  # Tangential std dev
        
        for section_id in measured_positions.keys():
            # Only use the first hole (index 0) from each section
            if len(measured_positions[section_id]) > 0:  # Make sure there's at least one hole
                hole_idx = 0
                x_obs, y_obs = measured_positions[section_id][hole_idx]
                
                mean_x, mean_y = model.forward(section_id, hole_idx)
                
                # Compute unit radial and tangential vectors
                r_hat = torch.tensor([mean_x, mean_y]) / torch.norm(torch.tensor([mean_x, mean_y]))
                t_hat = torch.tensor([-r_hat[1], r_hat[0]])  # Perpendicular to r_hat

                # Covariance matrix aligned with radial/tangential directions
                cov_matrix = (sigma_r**2) * torch.ger(r_hat, r_hat) + (sigma_t**2) * torch.ger(t_hat, t_hat)
                pyro.sample(
                    f"obs_{section_id}_{hole_idx}",
                    dist.MultivariateNormal(torch.tensor([mean_x, mean_y]), covariance_matrix=cov_matrix),
                    obs=torch.tensor([x_obs, y_obs])
                )


# Run MCMC sampling
def sampling(covariance_model="isotropic"):
    def model():
        return bayesian_model(covariance_model)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=200, num_chains=1)
    mcmc.run()

    # Extract posterior samples
    posterior_samples = mcmc.get_samples()

    # Select a hole for visualization
    hole_section = list(measured_positions.keys())[1]  # Select the first section
    hole_index = 0  # Select the first hole

    # Create empty arrays for posterior predictive samples
    num_samples = len(posterior_samples['r'])
    pred_x = np.zeros(num_samples)
    pred_y = np.zeros(num_samples)

    # Instantiate the model (same as in bayesian_model)
    model = AntikytheraModel(section_ids=measured_positions.keys(), 
                            covariance_type=covariance_model)

    # Generate posterior predictive samples manually
# Generate posterior predictive samples manually
    for i in range(num_samples):
        # Get parameters from posterior samples
        r_sample = posterior_samples['r'][i].item()
        N_sample = posterior_samples['N'][i].item()
        alpha_sample = posterior_samples[f'alpha_{hole_section}'][i].item()
        if covariance_model == "isotropic":
            sigma_sample = posterior_samples['sigma'][i].item()
        else:
            sigma_sample = {
                "r": posterior_samples['sigma_r'][i].item(),
                "t": posterior_samples['sigma_t'][i].item()}
        
        # Update model parameters correctly
        with torch.no_grad():
            # Convert Python floats to tensors before assigning
            model.r.copy_(torch.tensor(r_sample,dtype=torch.float64))
            model.N.copy_(torch.tensor(N_sample,dtype=torch.float64))
            alpha_param = getattr(model, f'alpha_{hole_section}')
            alpha_param.copy_(torch.tensor(alpha_sample, dtype=torch.float64))

        # Get predicted mean position
        mean_x, mean_y = model.forward(hole_section, hole_index)
        mean_tensor = torch.tensor([mean_x, mean_y], dtype=torch.float64)
        # Generate a sample from the observation distribution
        if covariance_model == "isotropic":
            cov_matrix = torch.eye(2, dtype=torch.float64) * sigma_sample**2
        else:
            dx = mean_x - measured_positions[hole_section][hole_index][0]
            dy = mean_y - measured_positions[hole_section][hole_index][1]

            r_mag = torch.sqrt(dx**2 + dy**2)
                    
            r_mag = torch.max(r_mag, torch.tensor(1e-10, dtype=torch.float64))
                    
            radial = torch.tensor([dx/r_mag, dy/r_mag], dtype=torch.float64)
            tangential = torch.tensor([-dy/r_mag, dx/r_mag], dtype=torch.float64)
                    
            rotation = torch.stack([radial, tangential], dim=1)
                    
            diag_cov = torch.diag(torch.tensor([sigma_sample["r"]**2, sigma_sample["t"]**2], dtype=torch.float64))
                    
            cov_matrix = torch.matmul(rotation, torch.matmul(diag_cov, rotation.t()))
            
        sample = dist.MultivariateNormal(
            mean_tensor, 
            covariance_matrix=cov_matrix
        ).sample()
        
        pred_x[i] = sample[0].item()
        pred_y[i] = sample[1].item()

    # Now plot as before
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_x, pred_y, alpha=0.5, label=f"Posterior Samples ({covariance_model})")
    plt.scatter(*measured_positions[hole_section][hole_index], color="red", label="Measured Position")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title(f"Posterior Predictive Distribution for Hole {hole_index} in Section {hole_section} ({covariance_model})")
    plt.savefig(f"graphs/posterior_predictive_{covariance_model}.png")
    plt.show()

def main():
    sampling("isotropic")
    sampling("anisotropic")

if __name__ == "__main__":
    main()