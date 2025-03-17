
import numpy as np 
import copy
import torch    

def convert_params_to_array(params, covariance_type="isotropic"):
    """
    Convert parameter dictionary to flat array for PyTorch differentiation.
    
    Args:
        params: Dictionary containing model parameters
        covariance_type: "isotropic" or "radial_tangential"
    
    Returns:
        param_array: Flattened array of parameters
        param_names: List of parameter names corresponding to array elements
    """
    param_array = []
    param_names = []
    
    # Add global parameters
    param_array.append(params['r'])
    param_names.append('r')
    
    param_array.append(params['N'])
    param_names.append('N')
    
    if covariance_type == "isotropic":
        param_array.append(params['sigma'])
        param_names.append('sigma')
    else:
        param_array.append(params['sigma_r'])
        param_names.append('sigma_r')
        
        param_array.append(params['sigma_t'])
        param_names.append('sigma_t')
    
    # Add section parameters
    for section in sorted(params['section_params'].keys()):
        for i, param_name in enumerate(['x0', 'y0', 'alpha']):
            param_array.append(params['section_params'][section][i])
            param_names.append(f'section_{section}_{param_name}')
    
    print(param_array)
    print(param_names)
            
    return np.array(param_array), param_names

def convert_array_to_params(param_array, param_names, template_params):
    """
    Convert flat parameter array back to parameter dictionary while preserving gradients.
    """
    params = {}
    
    # Check if we need to preserve gradients
    preserve_grad = torch.is_tensor(param_array) and param_array.requires_grad
    
    for i, name in enumerate(param_names):
        if name == 'r':
            params['r'] = param_array[i]  # Keep as tensor
        elif name == 'N':
            params['N'] = param_array[i]  # Keep as tensor
        elif name == 'sigma':
            params['sigma'] = param_array[i]  # Keep as tensor
        elif name == 'sigma_r':
            params['sigma_r'] = param_array[i]  # Keep as tensor
        elif name == 'sigma_t':
            params['sigma_t'] = param_array[i]  # Keep as tensor
        elif name.startswith('section_'):
            parts = name.split('_')
            section = int(parts[1])
            param = parts[2]
            
            # Initialize section_params if not already there
            if 'section_params' not in params:
                params['section_params'] = {}
            
            if section not in params['section_params']:
                params['section_params'][section] = [None, None, None]
            
            idx = 0
            if param == 'x0':
                idx = 0
            elif param == 'y0':
                idx = 1
            elif param == 'alpha':
                idx = 2
                
            params['section_params'][section][idx] = param_array[i]  # Keep as tensor
    
    return params