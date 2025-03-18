import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AntikytheraModel(nn.Module):
    """
    PyTorch module for the Antikythera mechanism model.
    This handles parameters properly within PyTorch's autograd framework.
    """
    def __init__(self, section_ids, covariance_type="isotropic"):
        super(AntikytheraModel, self).__init__()
        
        self.covariance_type = covariance_type
        self.section_ids = sorted(section_ids)
        
        self.r = nn.Parameter(torch.tensor(77.0, dtype=torch.float64))
        self.N = nn.Parameter(torch.tensor(354.0, dtype=torch.float64))
        
        if covariance_type == "isotropic":
            self.sigma = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        else:
            self.sigma_r = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            self.sigma_t = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        
    
        # Initialize
        section_params = {}
        for i, section in enumerate(self.section_ids):
            x0 = 80.0 + 0.5 * i
            y0 = 136.0
            alpha = -145.0 - i
            
            section_params[section] = [x0, y0, alpha]
        
        # Create parameters 
        for section in self.section_ids:
            x0, y0, alpha = section_params[section]
            setattr(self, f'x0_{section}', nn.Parameter(torch.tensor(x0, dtype=torch.float64)))
            setattr(self, f'y0_{section}', nn.Parameter(torch.tensor(y0, dtype=torch.float64)))
            setattr(self, f'alpha_{section}', nn.Parameter(torch.tensor(alpha, dtype=torch.float64)))
    
    def get_section_params(self, section):
        """Get parameters for a specific section."""
        x0 = getattr(self, f'x0_{section}')
        y0 = getattr(self, f'y0_{section}')
        alpha = getattr(self, f'alpha_{section}')
        return x0, y0, alpha
    
    def calculate_model_positions(self, hole_indices):
        """
        Calculate model-predicted positions for holes.
        
        Args:
            hole_indices: Dictionary mapping section indices to lists of hole indices
            
        Returns:
            Dictionary mapping section indices to tensors of coordinates
        """
        model_positions = {}
        
        for section, indices in hole_indices.items():
            if section not in self.section_ids:
                continue
                
            x0, y0, alpha = self.get_section_params(section)
            section_positions = []
            
            for i in indices:
                i_tensor = torch.tensor(float(i), dtype=torch.float64)
                
            
                phi = 2 * torch.pi * (i_tensor - 1) / self.N
               
                phi = phi + torch.deg2rad(alpha)
                
                x = x0 + self.r * torch.cos(phi)
                y = y0 + self.r * torch.sin(phi)
            
                pos = torch.stack([x, y])
                section_positions.append(pos)
            
            if section_positions:  
                model_positions[section] = torch.stack(section_positions)
        
        return model_positions
    
    def data_to_tensor(self, data):
        """
        Convert pandas DataFrame to dictionary of tensors.
        
        Args:
            data: DataFrame with measured data
            
        Returns:
            Dictionary mapping section IDs to tensors of coordinates
        """
        data_positions = {}
        for section, group in data.groupby("Section ID"):
            data_positions[section] = torch.tensor(
                [[x, y] for x, y in zip(group["Mean(X)"], group["Mean(Y)"])],
                dtype=torch.float64
            )
        return data_positions
    
    def negative_log_likelihood(self, data, hole_indices):
        """
        Calculate negative log-likelihood for optimization.
        
        Args:
            data: DataFrame with measured data
            hole_indices: Dictionary mapping section indices to lists of hole indices
            
        Returns:
            Negative log-likelihood as a scalar tensor
        """
       
        data_positions = self.data_to_tensor(data)
        
        model_positions = self.calculate_model_positions(hole_indices)
        
        # Initialize negative log-likelihood
        nll = torch.tensor(0.0, dtype=torch.float64)
        log_2pi = torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
        
        if self.covariance_type == "isotropic":
            cov_matrix = torch.eye(2, dtype=torch.float64) * (self.sigma**2)
            inv_cov = torch.inverse(cov_matrix)
            log_det = torch.log(torch.det(cov_matrix))
            
            for section in data_positions.keys():
                if section not in model_positions:
                    continue
                    
                data_pos = data_positions[section]
                model_pos = model_positions[section]
                
                for i in range(len(data_pos)):
                    diff = data_pos[i] - model_pos[i]
                    term = torch.dot(diff, torch.matmul(inv_cov, diff))
                    nll = nll + 0.5 * (term + log_det + 2 * log_2pi)
        
        else: 
            for section in data_positions.keys():
                if section not in model_positions:
                    continue
                    
                data_pos = data_positions[section]
                model_pos = model_positions[section]
                x0, y0, _ = self.get_section_params(section)
                
                for i in range(len(data_pos)):
                    # Vector from center to hole
                    dx = model_pos[i][0] - x0
                    dy = model_pos[i][1] - y0
                    r_mag = torch.sqrt(dx**2 + dy**2)
                    
                    r_mag = torch.max(r_mag, torch.tensor(1e-10, dtype=torch.float64))
                    
                    radial = torch.tensor([dx/r_mag, dy/r_mag], dtype=torch.float64)
                    tangential = torch.tensor([-dy/r_mag, dx/r_mag], dtype=torch.float64)
                    
                    rotation = torch.stack([radial, tangential], dim=1)
                    
                    diag_cov = torch.diag(torch.tensor([self.sigma_r**2, self.sigma_t**2], dtype=torch.float64))
                    
                    cov_matrix = torch.matmul(rotation, torch.matmul(diag_cov, rotation.t()))
                    
                    cov_matrix = cov_matrix + torch.eye(2, dtype=torch.float64) * 1e-10
                    
                    inv_cov = torch.inverse(cov_matrix)
                    log_det = torch.log(torch.det(cov_matrix))
                    
                    diff = data_pos[i] - model_pos[i]
                    
                    term = torch.dot(diff, torch.matmul(inv_cov, diff))
                    nll = nll + 0.5 * (term + log_det + 2 * log_2pi)
        
        return nll
    
    def to_dict(self):
        """
        Convert model parameters to dictionary format for compatibility.
        
        Returns:
            Dictionary containing model parameters
        """
        params = {
            'N': self.N.item(),
            'r': self.r.item(),
            'section_params': {}
        }
        
        if self.covariance_type == "isotropic":
            params['sigma'] = self.sigma.item()
        else:
            params['sigma_r'] = self.sigma_r.item()
            params['sigma_t'] = self.sigma_t.item()
        
        for section in self.section_ids:
            x0, y0, alpha = self.get_section_params(section)
            params['section_params'][section] = [x0.item(), y0.item(), alpha.item()]
        
        return params