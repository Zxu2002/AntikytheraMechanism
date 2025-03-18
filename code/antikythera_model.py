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
            self._log_sigma = nn.Parameter(torch.tensor(np.log(0.3), dtype=torch.float64))


        else:
            self._log_sigma_r = nn.Parameter(torch.tensor(np.log(0.3), dtype=torch.float64))
            self._log_sigma_t = nn.Parameter(torch.tensor(np.log(0.3), dtype=torch.float64))
        
    
        # Initialize
        section_params = {}
        for i, section in enumerate(self.section_ids):
            x0 = 80.0 + i
            y0 = 136.0
            alpha = -145.0 - i
            
            section_params[section] = [x0, y0, alpha]
        
        # Create parameters 
        for section in self.section_ids:
            x0, y0, alpha = section_params[section]
            setattr(self, f'x0_{section}', nn.Parameter(torch.tensor(x0, dtype=torch.float64)))
            setattr(self, f'y0_{section}', nn.Parameter(torch.tensor(y0, dtype=torch.float64)))
            setattr(self, f'alpha_{section}', nn.Parameter(torch.tensor(alpha, dtype=torch.float64)))

    @property
    def sigma_r(self):
        return torch.exp(self._log_sigma_r)
            
            
    @property
    def sigma_t(self):
        return torch.exp(self._log_sigma_t)
    
    @property
    def sigma(self):
        return torch.exp(self._log_sigma)
    
    @property
    def N(self):
        return self._N_float.round() 
    
    def forward(self, section, hole_index):
        """Predict hole position for a given section and index."""
        x0, y0, alpha = self.get_section_params(section)
        
        # Compute expected hole location
        theta = 2 * torch.pi * (hole_index - 1) / self.N 
        x_pred = x0 + self.r * torch.cos(theta + torch.deg2rad(alpha))
        y_pred = y0 + self.r * torch.sin(theta + torch.deg2rad(alpha))

        return x_pred, y_pred
    
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
                
            section_positions = []
            
            for i in indices:
                i_tensor = torch.tensor(float(i), dtype=torch.float64)
                
            
                x,y = self.forward(section, i_tensor)
            
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
def load_hole_data(filename):
    """
    Load the hole location data from the file.
    
    Args:
        filename: Path to the data file
        
    Returns:
        data: DataFrame with measured data
        measured_positions: Dictionary mapping section indices to lists of hole coordinates
        hole_indices: Dictionary mapping section indices to lists of hole indices
    """
    data = pd.read_csv(filename)
    
    # Convert measured positions to hole indices
    measured_positions = {}
    for section, group in data.groupby("Section ID"):
        measured_positions[section] = [np.array([x, y]) for x, y in zip(group["Mean(X)"], group["Mean(Y)"]) ]

    hole_indices = {}
    for section, positions in measured_positions.items():
        hole_indices[section] = list(range(1, len(positions) + 1))
    
    return data, measured_positions, hole_indices


def main():
    filename = "data/1-Fragment_C_Hole_Measurements.csv"
    data, measured_positions, hole_indices = load_hole_data(filename)

    section_id = [i for i in range(1,8)]
    model = AntikytheraModel(section_id)

    #plot the predicted hole positions
    plt.figure(figsize=(10, 10))
    unique_sections = data["Section ID"].unique()
    colors = plt.cm.get_cmap("tab10", len(unique_sections))
    section_colors = {}
    for i, section in enumerate(measured_positions.keys()):
        section_colors[section] = colors(i)
    for section, positions in measured_positions.items():
        color = section_colors.get(section, 'k')
        if positions is not None:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys, color=color, label=f'Section {section} (Measured)', marker='o', alpha=0.7)
    model_positions = model.calculate_model_positions(hole_indices)
    for section, positions in model_positions.items():
        if section not in measured_positions:
            continue
        color = section_colors.get(section, 'k')
        if positions is not None:
            xs = [float(x) for x in positions[:, 0].detach().numpy()]
            ys = [float(y) for y in positions[:, 1].detach().numpy()]
            plt.scatter(xs, ys, color=color, label=f'Section {section} (Predicted)', marker='x', alpha=0.5, s=100)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title("Predicted Hole Locations in the X-Y Plane")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig("graphs/predicted_hole_positions.png")
    plt.show()

    return 0 

if __name__ == "__main__":
    main()