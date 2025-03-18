import numpy as np
import pandas as pd
import torch
from params_tensor_conver import convert_params_to_array
from paper_model import log_likelihood

def torch_gradient(params, data, hole_indices, covariance_type="isotropic", device='cpu'):
    """
    Calculate gradient using PyTorch automatic differentiation.
    
    Parameters:
        - params: Dictionary of parameters
        - data: DataFrame with measured data
        - hole_indices: Dictionary mapping section indices to lists of hole indices
        - covariance_type: "isotropic" or "radial_tangential"
        - device: PyTorch device ('cpu' or 'cuda')
    
    Returns:
        Dictionary with gradients
    """
    # Convert parameters to array
    param_array, param_names = convert_params_to_array(params, covariance_type)
    
    # Convert to PyTorch tensor with gradient tracking
    param_tensor = torch.tensor(param_array, dtype=torch.float64, device=device, requires_grad=True)
    
    # Calculate log-likelihood
    log_like = log_likelihood(param_tensor, param_names, params, data, hole_indices, covariance_type, device)
    if param_tensor.grad is not None:
        param_tensor.grad.zero_()
    # Calculate gradient
    log_like.backward()
    grad_tensor = param_tensor.grad
    
    # Convert gradient tensor back to dictionary format
    if param_tensor.grad is None:
        print("WARNING: No gradient was calculated. Check for issues in the computational graph.")
        # Initialize zeros for the gradient to avoid errors
        grad_dict = {
            'r': 0.0,
            'N': 0.0,
            'section_params': {section: np.zeros(3) for section in params['section_params']}
        }
        
        if covariance_type == "isotropic":
            grad_dict['sigma'] = 0.0
        else:
            grad_dict['sigma_r'] = 0.0
            grad_dict['sigma_t'] = 0.0
            
        return grad_dict
    
    # Get the gradient tensor
    grad_tensor = param_tensor.grad
    print(f"Gradient tensor shape: {grad_tensor.shape}")
    
    # Convert gradient tensor back to dictionary format
    grad_dict = {
        'r': 0.0,
        'N': 0.0,
        'section_params': {section: np.zeros(3) for section in params['section_params']}
    }
    
    if covariance_type == "isotropic":
        grad_dict['sigma'] = 0.0
    else:
        grad_dict['sigma_r'] = 0.0
        grad_dict['sigma_t'] = 0.0
    
    # Fill in gradient values from tensor
    for i, name in enumerate(param_names):
        if name == 'r':
            grad_dict['r'] = grad_tensor[i].item()
        elif name == 'N':
            grad_dict['N'] = grad_tensor[i].item()
        elif name == 'sigma':
            grad_dict['sigma'] = grad_tensor[i].item()
        elif name == 'sigma_r':
            grad_dict['sigma_r'] = grad_tensor[i].item()
        elif name == 'sigma_t':
            grad_dict['sigma_t'] = grad_tensor[i].item()
        elif name.startswith('section_'):
            parts = name.split('_')
            section = int(parts[1])
            param = parts[2]
            
            idx = 0
            if param == 'x0':
                idx = 0
            elif param == 'y0':
                idx = 1
            elif param == 'alpha':
                idx = 2
                
            grad_dict['section_params'][section][idx] = grad_tensor[i].item()
    
    return grad_dict


def test_torch_gradient():
    """
    Simple unit test for the PyTorch automatic differentiation gradient.
    """
    # Create a simple test case with just one parameter for testing
    test_params = {
        'r': 10.0,
        'N': 100.0,
        'sigma': 1.0,
        'section_params': {
            1: [0.0, 0.0, 0.0]
        }
    }
    
    # Create minimal data that will definitely intersect with your model
    data_dict = {
        'Section ID': [1],
        'Mean(X)': [10.0],
        'Mean(Y)': [0.0]
    }
    test_data = pd.DataFrame(data_dict)
    
    # Create hole indices that match your data
    test_hole_indices = {1: [1]}
    
    # Convert parameters to tensor with requires_grad=True
    param_array, param_names = convert_params_to_array(test_params)
    param_tensor = torch.tensor(param_array, dtype=torch.float64, requires_grad=True)
    
    # Calculate log-likelihood directly
    log_like = log_likelihood(param_tensor, param_names, test_params, test_data, test_hole_indices)
    print(f"Log-likelihood value: {log_like.item()}")
    
    # Calculate gradient
    log_like.backward()
    grad = param_tensor.grad
    
    if grad is None:
        print("Still no gradient! Let's debug more deeply.")
        return None
    
    print("Gradient calculated successfully!")
    
    # Convert back to dictionary format
    grad_dict = {
        'r': grad[0].item(),
        'N': grad[1].item(),
        'sigma': grad[2].item(),
        'section_params': {
            1: [grad[3].item(), grad[4].item(), grad[5].item()]
        }
    }
    
    print(f"Gradient values: r={grad_dict['r']:.4f}, N={grad_dict['N']:.4f}, sigma={grad_dict['sigma']:.4f}")
    print(f"Section params: x0={grad_dict['section_params'][1][0]:.4f}, y0={grad_dict['section_params'][1][1]:.4f}, alpha={grad_dict['section_params'][1][2]:.4f}")
    
    return grad_dict

def main():
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
    convert_params_to_array(initial_params)

    # Run a simple test
    print("Testing PyTorch automatic differentiation for the Antikythera mechanism model")
    print(test_torch_gradient())
    
if __name__ == "__main__":
    main()
