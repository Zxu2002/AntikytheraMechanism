import torch
import matplotlib.pyplot as plt
from antikythera_model import AntikytheraModel, load_hole_data



def optimize_model(data, hole_indices, model_type="isotropic", learning_rate=0.0001, num_iterations=1000):
    """
    Find the maximum likelihood parameters using PyTorch optimization.
    
    Args:
        data: DataFrame with measured data
        hole_indices: Dictionary mapping section indices to lists of hole indices
        model_type: "isotropic" or "radial_tangential"
        learning_rate: Learning rate for optimizer
        num_iterations: Number of iterations to perform
        
    Returns:
        model: Optimized AntikytheraModel
        losses: List of loss values during optimization
    """
    section_ids = data["Section ID"].unique()
    
    model = AntikytheraModel(section_ids, covariance_type=model_type)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    # Optimization loop
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        loss = model.negative_log_likelihood(data, hole_indices)
        
        if not torch.isfinite(loss):
            print(f"Warning: Loss is not finite at iteration {i}")
            break

        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if torch.isfinite(torch.tensor(total_norm)) and total_norm > 0:
            optimizer.step()

        else:
            print(f"Skipping optimizer step due to gradients: norm = {total_norm}")
        
        losses.append(loss.item())
    
    return model, losses

def plot_optimization(losses, title="Optimization Progress", save_path=None):
    """
    Plot the optimization progress.
    
    Args:
        losses: List of loss values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_results(data, measured_positions, model, hole_indices, title="Antikythera Mechanism Calendar Ring Hole Positions", save_path=None):
    """
    Plot the measured hole positions and model predictions.
    
    Args:
        data: DataFrame with measured data
        measured_positions: Dictionary mapping section indices to measured hole positions
        model: AntikytheraModel with optimized parameters
        hole_indices: Dictionary mapping section indices to lists of hole indices
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 10))
    
    # Get unique sections and assign colors
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
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    # Load the data
    data_filename = "data/1-Fragment_C_Hole_Measurements.csv"
    data, measured_positions, hole_indices = load_hole_data(data_filename)
    
    # Optimize isotropic model
    print("Optimizing isotropic covariance model...")
    model_iso, losses_iso = optimize_model(
        data, 
        hole_indices, 
        model_type="isotropic", 
        learning_rate=0.001,
        num_iterations=1000
    )
    
    print("\nOptimized parameters (isotropic):")
    params_iso = model_iso.to_dict()
    print(f"N = {params_iso['N']:.2f}")
    print(f"r = {params_iso['r']:.2f} mm")
    print(f"sigma = {params_iso['sigma']:.4f}")
    
    for section, params in params_iso['section_params'].items():
        print(f"Section {section}: x0 = {params[0]:.2f}, y0 = {params[1]:.2f}, alpha = {params[2]:.2f}")
    
    plot_optimization(losses_iso, title="Isotropic Model Optimization", save_path="graphs/isotropic_optimization.png")
    
    # Plot the results
    plot_results(
        data, 
        measured_positions, 
        model_iso, 
        hole_indices, 
        title="Calendar Ring Hole Positions: Isotropic Model",
        save_path="graphs/isotropic_results.png"
    )
    
    # Optionally optimize radial/tangential model
    print("\nOptimizing radial/tangential covariance model...")
    model_rt, losses_rt = optimize_model(
        data, 
        hole_indices, 
        model_type="radial_tangential", 
        learning_rate=0.001,
        num_iterations=1000
    )
    
    # Print optimized parameters
    print("\nOptimized parameters (radial/tangential):")
    params_rt = model_rt.to_dict()
    print(f"N = {params_rt['N']:.2f}")
    print(f"r = {params_rt['r']:.2f} mm")
    print(f"sigma_r = {params_rt['sigma_r']:.4f}")
    print(f"sigma_t = {params_rt['sigma_t']:.4f}")
    
    for section, params in params_rt['section_params'].items():
        print(f"Section {section}: x0 = {params[0]:.2f}, y0 = {params[1]:.2f}, alpha = {params[2]:.2f}")
    
    plot_optimization(losses_rt, title="Radial/Tangential Model Optimization", save_path="graphs/radial_tangential_optimization.png")
    
    plot_results(
        data, 
        measured_positions, 
        model_rt, 
        hole_indices, 
        title="Calendar Ring Hole Positions: Radial/Tangential Model",
        save_path="graphs/radial_tangential_results.png"
    )

if __name__ == "__main__":
    main()