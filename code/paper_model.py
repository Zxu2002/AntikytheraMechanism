import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal


# def model(N,x0,y0,alpha):
#     phi = np.array([])
#     for i in range(N):
#         temp = np.array([])
#         for j in range(N):
#             temp = np.append(temp,2*np.pi*(i-1)/N + alpha[j])
              
#         phi = np.append(phi,temp)
    

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
    return data 
  

def calculate_model_positions(params, hole_indices):
    """
    Calculate the model-predicted positions for all holes.
    
    Args:
        params: Dictionary containing model parameters (N, r, section_params)
        hole_indices: Dictionary mapping section indices to lists of hole indices
        
    Returns:
        Dictionary mapping section indices to lists of model-predicted (x,y) coordinates
    """
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
            # Calculate the angular position
            phi = 2 * np.pi * (i - 1) / N + alpha
            
            # Calculate the Cartesian coordinates
            x = x0 + r * np.cos(phi)
            y = y0 + r * np.sin(phi)
            
            section_positions.append((x, y))
            
        model_positions[section_idx] = section_positions
    
    return model_positions

def compute_model_predictions(theta, data_indices):
    """
    Compute the model predictions for hole locations based on parameter vector theta.
    
    Parameters:
        - theta: 
        The parameter vector which includes:
        - r: radius of the circle
        - N: number of holes in the original complete ring
        - sigma parameters (either single sigma or sigma_r and sigma_t)
        - For each section j: x_j, y_j, alpha_j (translation and rotation)
    
    data_indices: Indices of holes in the original configuration for each measured hole
    
    Returns:
        - model_positions: Predicted (x,y) positions of each hole based on the model
    """
    # Extract parameters from theta
    r = theta[0]                # radius of the ring
    N = theta[1]                # number of holes in the complete ring
    
    # The remaining parameters depend on how many sections there are
    # Assuming 8 sections (j ∈ {0, 1, 2, ..., 7})
    num_sections = 8
    
    # Check if we're using isotropic or radial/tangential covariance
    if len(theta) == 2 + 1 + 3*num_sections:  # isotropic case
        sigma_offset = 3
    else:  # radial/tangential case
        sigma_offset = 4
    
    # Calculate the original positions (before transformations)
    # Original positions on the circle
    angle_step = 2 * np.pi / N
    original_positions = np.zeros((len(data_indices), 2))
    
    for i, idx in enumerate(data_indices):
        angle = idx * angle_step
        original_positions[i, 0] = r * np.cos(angle)
        original_positions[i, 1] = r * np.sin(angle)
    
    # Apply transformations for each section
    model_positions = np.zeros_like(original_positions)
    section_indices = np.zeros(len(data_indices), dtype=int)
    
    # Assuming section_indices is given for each hole in the data
    # For each section, apply the transformation
    for j in range(num_sections):
        # Extract transformation parameters for this section
        x_j = theta[sigma_offset + 3*j]
        y_j = theta[sigma_offset + 3*j + 1]
        alpha_j = theta[sigma_offset + 3*j + 2]
        
        # Get holes belonging to this section
        section_mask = (section_indices == j)
        
        if np.any(section_mask):
            # Get the original positions for this section
            section_orig_pos = original_positions[section_mask]
            
            # Apply rotation
            rot_matrix = np.array([
                [np.cos(alpha_j), -np.sin(alpha_j)],
                [np.sin(alpha_j), np.cos(alpha_j)]
            ])
            
            # Apply rotation and translation
            rotated = np.dot(section_orig_pos, rot_matrix.T)
            translated = rotated + np.array([x_j, y_j])
            
            # Store the transformed positions
            model_positions[section_mask] = translated
    
    return model_positions

def log_likelihood(theta, data_positions, data_indices, section_indices, covariance_type="isotropic"):
    """
    Calculate the log-likelihood given the parameters and data.
    
    Parameters:
    -----------
    theta : array-like
        The parameter vector
    data_positions : ndarray
        Measured (x,y) positions of each hole
    data_indices : array-like
        Indices of holes in the original configuration
    section_indices : array-like
        Section index for each hole
    covariance_type : str
        Type of covariance matrix: "isotropic" or "radial_tangential"
    
    Returns:
    --------
    log_likelihood : float
        The log-likelihood value
    """
    # Extract parameters
    r = theta[0]
    N = theta[1]
    
    if covariance_type == "isotropic":
        sigma = theta[2]
        # Isotropic covariance matrix
        cov_matrix = np.eye(2) * sigma**2
    else:  # radial_tangential
        sigma_r = theta[2]
        sigma_t = theta[3]
        # We'll construct the covariance matrix for each hole separately
    
    # Compute model predictions
    model_positions = compute_model_predictions(theta, data_indices)
    
    # Calculate log-likelihood
    log_like = 0.0
    
    for i in range(len(data_positions)):
        data_pos = data_positions[i]
        model_pos = model_positions[i]
        
        if covariance_type == "isotropic":
            # Use the same covariance matrix for all holes
            mvn = multivariate_normal(mean=model_pos, cov=cov_matrix)
            log_like += mvn.logpdf(data_pos)
        else:
            # For radial/tangential, we need to construct the covariance matrix
            # based on the position of each hole
            angle = np.arctan2(model_pos[1], model_pos[0])
            # Radial and tangential unit vectors
            radial = np.array([np.cos(angle), np.sin(angle)])
            tangential = np.array([-np.sin(angle), np.cos(angle)])
            
            # Construct rotation matrix to transform from (radial, tangential) to (x, y)
            rotation = np.column_stack((radial, tangential))
            
            # Diagonal covariance in (radial, tangential) coordinates
            diag_cov = np.diag([sigma_r**2, sigma_t**2])
            
            # Transform to (x, y) coordinates
            cov_matrix = rotation @ diag_cov @ rotation.T
            
            # Calculate likelihood
            mvn = multivariate_normal(mean=model_pos, cov=cov_matrix)
            log_like += mvn.logpdf(data_pos)
    
    return log_like

def negative_log_likelihood(theta, data_positions, data_indices, section_indices, covariance_type="isotropic"):
    """
    Calculate the negative log-likelihood for optimization purposes.
    """
    return -log_likelihood(theta, data_positions, data_indices, section_indices, covariance_type)


def plot_holes(data, measured_positions, model_positions=None):
    """
    Plot the measured hole positions and optionally the model predictions.
    
    Args:
        measured_positions: Dictionary mapping section indices to measured hole positions
        model_positions: Dictionary mapping section indices to model-predicted hole positions
        section_colors: Dictionary mapping section indices to colors
    """
    plt.figure(figsize=(10, 10))
    
    unique_sections = data["Section ID"].unique()
    colors = plt.cm.get_cmap("tab10", len(unique_sections)) 

        # Default colors for different sections
        
    section_colors = {}
    for i, section in enumerate(measured_positions.keys()):
        section_colors[section] = colors(i)
    
    # Plot measured positions
    for section, positions in measured_positions.items():
        color = section_colors.get(section, 'k')
        if positions:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys, color=color, label=f'Section {section} (Measured)', marker='o', alpha=0.7)
    
    # Plot model positions if provided
    if model_positions:
        for section, positions in model_positions.items():
            if section not in measured_positions:
                continue
                
            color = section_colors.get(section, 'k')
            if positions:
                xs, ys = zip(*positions)
                plt.scatter(xs, ys, color=color, marker='x', alpha=0.5, s=100)
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Antikythera Mechanism Calendar Ring Hole Positions')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig('graphs/predicted_hole_positions.png')
    plt.show()

def main():
    # Load the hole data
    file_path = "data/1-Fragment_C_Hole_Measurements.csv"  
    data = load_hole_data(file_path)
    print(data.head)
    # Convert measured positions to hole indices
    measured_positions = {}
    for section, group in data.groupby("Section ID"):
        measured_positions[section] = list(zip(group["Mean(X)"], group["Mean(Y)"]))

    hole_indices = {}
    for section, positions in measured_positions.items():
        hole_indices[section] = list(range(1, len(positions) + 1))
    

    initial_params = {
        'N': 354.0,  # Initial guess for number of holes
        'r': 77.0,   # Initial guess for radius (mm)
        'section_params': {
            # Initial guesses for section parameters (x0, y0, alpha)
            # You'll need to adjust these based on the data
            1: (80.0, 136.0, -2.54),
            2: (80.0, 136.0, -2.54),
            3: (80.0, 136.0, -2.54),
            5: (81.0, 136.0, -2.56),
            6: (81.5, 136.0, -2.56),
            7: (83.0, 136.5, -2.58),
        }
    }
    
    # Calculate model positions with initial parameters
    model_positions = calculate_model_positions(initial_params, hole_indices)
    
    # Plot the results
    plot_holes(data, measured_positions, model_positions)


if __name__ == "__main__":
    main()