#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import necessary functions from provided modules
from paper_model import negative_log_likelihood, calculate_model_positions, load_hole_data, plot_holes
from params_tensor_conver import convert_params_to_array, convert_array_to_params

def optimize_model(data, hole_indices, model_type):
    """
    Find the maximum likelihood parameters by minimizing the negative log-likelihood.
    """
    # Load initial parameters
    initial_params = {
        'N': 354.0,  # Initial guess for number of holes
        'r': 77.0,   # Initial guess for radius (mm)
        'section_params': {  # Initial guesses for section transformations
            1: [80.0, 136.0, -145.0],
            2: [80.0, 136.0, -145.0],
            3: [80.0, 136.0, -145.0],
            4: [80.5, 136.0, -146.0],
            5: [81.0, 136.0, -146.0],
            6: [81.5, 136.0, -146.0],
            7: [83.0, 136.5, -147.0],
        },
        "sigma": 0.5,  # Only for isotropic
        "sigma_r": 0.2,  # Only for radial/tangential
        "sigma_t": 0.5,  # Only for radial/tangential
    }

    # Convert to numerical format
    param_array, param_names = convert_params_to_array(initial_params, covariance_type=model_type)
    params_tensor = params_tensor = torch.nn.Parameter(torch.tensor(param_array, dtype=torch.float32, requires_grad=True))

    # Define objective function
    # def objective(params_tensor):
    #     nll_value = negative_log_likelihood(params_tensor, param_names, initial_params, data, hole_indices, covariance_type=model_type)
        
    #     # Ensure nll_value is a scalar tensor for PyTorch optimization
    #     if isinstance(nll_value, torch.Tensor):
    #         return nll_value  # Return as PyTorch tensor to allow backprop

    # Optimize using BFGS
    optimizer = torch.optim.Adam([params_tensor], lr=0.005)  # Adjust learning rate if needed

    # Optimization loop
    num_iterations = 1000  # Adjust iterations as needed
    for i in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        loss = negative_log_likelihood(params_tensor, param_names, initial_params, data, hole_indices, covariance_type=model_type)

        loss.backward()  
        print("Gradients:", params_tensor.grad)
        optimizer.step() 

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")
            

    # Convert optimized parameters back to dictionary form
    opt_params = convert_array_to_params(params_tensor.detach().numpy(), param_names, initial_params)
    return opt_params

def plot_results(data, measured_positions, model_positions_iso, model_positions_rt):
    """
    Plot measured hole positions along with maximum likelihood predictions.
    """
    plot_holes(data, measured_positions, model_positions_iso, fig_title='Calendar Ring Hole Positions: Isotropic',fig_save_path='graphs/mle_iso_predicted_hole_positions.png')
    plot_holes(data, measured_positions, model_positions_rt, fig_title='Calendar Ring Hole Positions: Radial Tangential',fig_save_path='graphs/mle_rt_predicted_hole_positions.png')


def main():
    # Load the data
    data_filename = "data/1-Fragment_C_Hole_Measurements.csv"
    data, measured_positions, hole_indices = load_hole_data(data_filename)

    # Find MLE for isotropic model
    print("Optimizing isotropic covariance model...")
    ml_params_iso = optimize_model(data, hole_indices, model_type="isotropic")
    print(ml_params_iso)
    # Find MLE for radial/tangential model
    print("Optimizing radial/tangential covariance model...")
    ml_params_rt = optimize_model(data, hole_indices, model_type="radial_tangential")
    print(ml_params_rt)
    # Compute predicted hole positions for each model
    model_positions_iso = calculate_model_positions(ml_params_iso, hole_indices)
    model_positions_rt = calculate_model_positions(ml_params_rt, hole_indices)
   
    # Plot the results
    plot_results(data, measured_positions, model_positions_iso, model_positions_rt)

if __name__ == "__main__":
    main()
