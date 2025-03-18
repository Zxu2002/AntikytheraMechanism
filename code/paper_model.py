import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import torch
from params_tensor_conver import convert_params_to_array, convert_array_to_params


    

def load_hole_data(filename):
    """
    Load the hole location data from the file.
    
    Args:
        filename: Path to the data file
        
    Returns:
        Dictionary mapping section indices to lists of hole coordinates
        List of section indices
    """
    # This is a placeholder implementation - adapt based on actual data format
    data = pd.read_csv(filename)
    
    # Convert measured positions to hole indices
    measured_positions = {}
    for section, group in data.groupby("Section ID"):
        measured_positions[section] = [np.array([x, y]) for x, y in zip(group["Mean(X)"], group["Mean(Y)"]) ]

    hole_indices = {}
    for section, positions in measured_positions.items():
        hole_indices[section] = list(range(1, len(positions) + 1))
    
    return data , measured_positions, hole_indices
  

def calculate_model_positions(params, hole_indices):
    model_positions = {}
    N = params['N']
    r = params['r']
    section_params = params['section_params']
    
    for section_idx, indices in hole_indices.items():
        if section_idx not in section_params:
            continue
            
        x0, y0, alpha = section_params[section_idx]
        section_positions = []

        for i in indices:
            # Use existing tensors directly if they are tensors
            # Otherwise convert to tensor with requires_grad=True if input tensor had requires_grad=True
            requires_grad = any(torch.is_tensor(t) and t.requires_grad for t in [N, r, x0, y0, alpha])
            i_float = float(i)
            i_tensor = torch.tensor(i_float, dtype=torch.float64)
            N_tensor = N if torch.is_tensor(N) else torch.tensor(float(N), dtype=torch.float64)
            r_tensor = r if torch.is_tensor(r) else torch.tensor(float(r), dtype=torch.float64)
            x0_tensor = x0 if torch.is_tensor(x0) else torch.tensor(float(x0), dtype=torch.float64)
            y0_tensor = y0 if torch.is_tensor(y0) else torch.tensor(float(y0), dtype=torch.float64)
            alpha_tensor = alpha if torch.is_tensor(alpha) else torch.tensor(float(alpha), dtype=torch.float64)
            
            # Calculate the angular position (in radians)
            phi = 2 * torch.pi * (i_tensor - 1) / N_tensor
            
            # Add rotation angle
            phi = phi + torch.deg2rad(alpha_tensor)
            
            # Calculate coordinates 
            x = x0_tensor + r_tensor * torch.cos(phi)
            y = y0_tensor + r_tensor * torch.sin(phi)
            
            # Ensure they're combined as a tensor
            pos = torch.stack([x, y])
            section_positions.append(pos)
        
        if section_positions:  # Check if there are positions to convert
            model_positions[section_idx] = torch.stack(section_positions)

    return model_positions



def data_for_torch(data, device='cpu'):
    """
    Convert pandas DataFrame to PyTorch tensors.
    
    Args:
        data: DataFrame with measured data
        device: PyTorch device ('cpu' or 'cuda')
    
    Returns:
        Dictionary mapping section IDs to tensors of coordinates
    """
    data_positions = {}
    for section, group in data.groupby("Section ID"):
        data_positions[section] = torch.tensor(
            [[x, y] for x, y in zip(group["Mean(X)"], group["Mean(Y)"])],
            dtype=torch.float64,
            device=device
        )
    return data_positions




def log_likelihood(params_tensor, param_names, template_params, data, hole_indices, covariance_type="isotropic", device='cpu'):
    """
    PyTorch-compatible log-likelihood function.
    
    Args:
        params_tensor: PyTorch tensor of flattened parameters
        param_names: List of parameter names
        template_params: Template parameter dictionary
        data: DataFrame with measured data
        hole_indices: Dictionary mapping section indices to lists of hole indices
        covariance_type: "isotropic" or "radial_tangential"
        device: PyTorch device ('cpu' or 'cuda')
    
    Returns:
        log_like: Log-likelihood value as a PyTorch tensor
    """
    # Convert flat tensor to parameter dictionary
    if not torch.is_tensor(params_tensor):
        params_tensor = torch.tensor(params_tensor, dtype=torch.float64, device=device, requires_grad=True)
    if not params_tensor.requires_grad:
        params_tensor.requires_grad_(True)

    params = convert_array_to_params(params_tensor, param_names, template_params)
    # Convert data to PyTorch tensors
    data_positions = data_for_torch(data, device)
    
    # Get model predictions
    model_positions = calculate_model_positions(params, hole_indices)
    
    # Initialize log-likelihood
    log_like = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    log_2pi = torch.log(torch.tensor(2 * np.pi, dtype=torch.float64, device=device))
    if covariance_type == "isotropic":
        # For isotropic case
        sigma = params["sigma"]
        cov_matrix = torch.eye(2, dtype=torch.float64, device=device) * (sigma**2)
        inv_cov = torch.inverse(cov_matrix)
        log_det = torch.log(torch.det(cov_matrix))
        
        for section in data_positions.keys():
            if section not in model_positions:
                continue
                
            data_pos = data_positions[section]
            model_pos = model_positions[section]

            # Calculate multivariate normal log PDF for each hole
            for i in range(len(data_pos)):
                diff = data_pos[i] - model_pos[i]
                term = torch.dot(diff, torch.matmul(inv_cov, diff))
                log_like = log_like -0.5 * (term + log_det + 2 * log_2pi)
    
    else:  # radial_tangential case
        
        sigma_r = params["sigma_r"]
        sigma_t = params["sigma_t"]
        for section in data_positions.keys():
            if section not in model_positions:
                continue
                
            data_pos = data_positions[section]
            model_pos = model_positions[section]
            x0 = params['section_params'][section][0]
            y0 = params['section_params'][section][1]
            
            # Calculate log-likelihood for each hole
            for i in range(len(data_pos)):
                # Vector from center to hole
                dx = model_pos[i][0] - x0
                dy = model_pos[i][1] - y0
                r_mag = torch.sqrt(dx**2 + dy**2)
                
                # Avoid division by zero 
                r_mag = torch.maximum(r_mag, torch.tensor(1e-10, dtype=torch.float64, device=device))
                
                # Radial and tangential unit vectors
                radial = torch.tensor([dx/r_mag, dy/r_mag], dtype=torch.float64, device=device)
                tangential = torch.tensor([-dy/r_mag, dx/r_mag], dtype=torch.float64, device=device)
                
                # rotation matrix
                rotation = torch.stack([radial, tangential], dim=1)
                
                # covariance matrix in (r, t) coordinates
                diag_cov = torch.diag(torch.tensor([sigma_r**2, sigma_t**2], dtype=torch.float64, device=device))
                
                # Transform to (x, y) coordinates
                cov_matrix = torch.matmul(rotation, torch.matmul(diag_cov, rotation.t()))
                
                # Avoid singular matrices 
                cov_matrix = cov_matrix + torch.eye(2, dtype=torch.float64, device=device) * 1e-10
                
                # inverse and determinant
                inv_cov = torch.inverse(cov_matrix)
                log_det = torch.log(torch.det(cov_matrix))
                
                diff = data_pos[i] - model_pos[i]
                
                term = torch.dot(diff, torch.matmul(inv_cov, diff))
                log_like = log_like -0.5 * (term + log_det + 2 * log_2pi)

    if len(data) == 0 or all(section not in model_positions for section in data_positions.keys()):
        # Create a simple differentiable expression based on params
        log_like = -0.5 * torch.sum(params_tensor ** 2)
        log_like.backward(retain_graph=True)
        print(f"Gradient norm: {params_tensor.grad.norm() if params_tensor.grad is not None else 'No Gradients'}")

        return log_like
    
    log_like.backward(retain_graph=True)
    print(f"Gradient norm: {params_tensor.grad.norm() if params_tensor.grad is not None else 'No Gradients'}")

    return log_like

def negative_log_likelihood(params_tensor, param_names, theta, data, hole_indices, covariance_type="isotropic"):
    """
    Calculate the negative log-likelihood for optimization purposes.
    """
    return -log_likelihood(params_tensor, param_names, theta, data, hole_indices, covariance_type=covariance_type)

def tensor_to_list(tensor):
    if torch.is_tensor(tensor):
        return tensor.tolist()
    return tensor

def plot_holes(data, measured_positions, model_positions, fig_title = 'Antikythera Mechanism Calendar Ring Hole Positions',fig_save_path='graphs/predicted_hole_positions.png'):
    """
    Plot the measured hole positions and optionally the model predictions.
    
    Args:
        measured_positions: Dictionary mapping section indices to measured hole positions
        model_positions: Dictionary mapping section indices to model-predicted hole positions
        section_colors: Dictionary mapping section indices to colors
    """
    plt.figure(figsize=(10, 10))
    
    unique_sections = data["Section ID"].unique()
    colors = plt.get_cmap("tab10", len(unique_sections))

        
    section_colors = {}
    for i, section in enumerate(measured_positions.keys()):
        section_colors[section] = colors(i)
    
    # Plot measured positions
    for section, positions in measured_positions.items():
        color = section_colors.get(section, 'k')
        if positions is not None:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys, color=color, label=f'Section {section} (Measured)', marker='o', alpha=0.7)
    
    # Plot model positions if provided
    
    for section, positions in model_positions.items():
        if section not in measured_positions:
            continue
                
        color = section_colors.get(section, 'k')
        if positions is not None:
            xs = [float(x) for x in tensor_to_list(positions[:, 0])]
            ys = [float(y) for y in tensor_to_list(positions[:, 1])]
            plt.scatter(xs, ys, color=color, label = f'Section {section} (Predicted)', marker='x', alpha=0.5, s=100)
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(fig_title)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(fig_save_path)
    plt.show()

def main():
    # Load the hole data
    file_path = "data/1-Fragment_C_Hole_Measurements.csv"  
    data, measured_positions, hole_indices = load_hole_data(file_path)

    initial_params = {
        'N': 354.0,  # Initial guess for number of holes
        'r': 77.0,   # Initial guess for radius (mm)
        'section_params': {
            # Initial guesses for section parameters (x0, y0, alpha)
            1: [80.0, 136.0, -145.0],
            2: [80.0, 136.0, -145.0],
            3: [80.0, 136.0, -145.0],
            4: [80.5, 136.0, -146.0],
            5: [81.0, 136.0, -146.0],
            6: [81.5, 136.0, -146.0],
            7: [83.0, 136.5, -147.0],
        }, 
        "sigma": 1.0,  
        "sigma_r": 1.0,  
        "sigma_t": 1.0,  
    }
    params_tensor, param_names = convert_params_to_array(initial_params)
    
    model_positions = calculate_model_positions(initial_params, hole_indices)
    # print(model_positions)
    plot_holes(data, measured_positions, model_positions)

    log_likelihood(params_tensor, param_names, initial_params, data, hole_indices, covariance_type="isotropic")

    
if __name__ == "__main__":
    main()