import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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