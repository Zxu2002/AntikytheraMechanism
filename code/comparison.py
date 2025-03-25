import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats

from antikythera_model import load_hole_data
from optimize_antikythera import optimize_model

# Load the data
data_filename = "data/1-Fragment_C_Hole_Measurements.csv"
data, measured_positions, hole_indices = load_hole_data(data_filename)

def confidence_ellipse(ax, x, y, cov, n_std=1.0, **kwargs):
    """Draw a covariance ellipse on the given axes."""
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(x, y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def compare_models():
    """Compare isotropic and radial/tangential models visually and quantitatively."""
    # Optimize both models
    print("Optimizing isotropic model...")
    model_iso, _ = optimize_model(data, hole_indices, model_type="isotropic", 
                                 learning_rate=0.001, num_iterations=500)
    
    print("Optimizing radial/tangential model...")
    model_rt, _ = optimize_model(data, hole_indices, model_type="radial_tangential", 
                                learning_rate=0.001, num_iterations=500)
    
    # Extract model parameters
    params_iso = model_iso.to_dict()
    params_rt = model_rt.to_dict()
    print(params_iso)
    print("--------------------")
    print(params_rt)
    
    # Get model predictions
    model_pos_iso = model_iso.calculate_model_positions(hole_indices)
    model_pos_rt = model_rt.calculate_model_positions(hole_indices)
    
    # --- VISUALIZATION 1: Error ellipses comparison ---
    plt.figure(figsize=(12, 6))
    
    # Choose a section with good data
    section_id = 2
    
    # Subplot for isotropic model
    plt.subplot(1, 2, 1)
    for section, positions in measured_positions.items():
        if section == section_id:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys, color='blue', label='Measured', marker='o')
    
    if section_id in model_pos_iso:
        positions = model_pos_iso[section_id]
        xs = [float(x) for x in positions[:, 0].detach().numpy()]
        ys = [float(y) for y in positions[:, 1].detach().numpy()]
        plt.scatter(xs, ys, color='red', label='Predicted', marker='x')
        
        # Add covariance ellipses
        sigma_iso = params_iso['sigma']
        cov_iso = np.eye(2) * sigma_iso**2
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            confidence_ellipse(plt.gca(), x, y, cov_iso, n_std=2.0, 
                              edgecolor='red', alpha=0.3, linewidth=2)
    
    plt.title(f"Isotropic Model (Section {section_id})")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot for radial/tangential model
    plt.subplot(1, 2, 2)
    for section, positions in measured_positions.items():
        if section == section_id:
            xs, ys = zip(*positions)
            plt.scatter(xs, ys, color='blue', label='Measured', marker='o')
    
    if section_id in model_pos_rt:
        positions = model_pos_rt[section_id]
        xs = [float(x) for x in positions[:, 0].detach().numpy()]
        ys = [float(y) for y in positions[:, 1].detach().numpy()]
        plt.scatter(xs, ys, color='green', label='Predicted', marker='x')
        
        # Get center coordinates
        x0, y0, _ = model_rt.get_section_params(section_id)
        x0, y0 = float(x0), float(y0)
        
        # Add covariance ellipses
        sigma_r = params_rt['sigma_r']
        sigma_t = params_rt['sigma_t']
        
        for i, (x, y) in enumerate(zip(xs, ys)):
            # Vector from center to hole
            dx = x - x0
            dy = y - y0
            r_mag = np.sqrt(dx**2 + dy**2)
            
            # Change of coordiante 
            r_hat = np.array([dx/r_mag, dy/r_mag])
            t_hat = np.array([-dy/r_mag, dx/r_mag])
            
            # Covariance matrix
            rotation = np.column_stack((r_hat, t_hat))
            diag_cov = np.diag([sigma_r**2, sigma_t**2])
            cov_rt = rotation @ diag_cov @ rotation.T
            
            confidence_ellipse(plt.gca(), x, y, cov_rt, n_std=2.0, 
                              edgecolor='green', alpha=0.3, linewidth=2)
    
    plt.title(f"Radial/Tangential Model (Section {section_id})")
    plt.xlabel("X Position (mm)")
    plt.ylabel("Y Position (mm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("graphs/covariance_comparison.png")
    
    # --- VISUALIZATION 2: Residual distributions ---
    plt.figure(figsize=(8, 6))
    
    # Compute residuals for both models
    residuals_iso = []
    residuals_rt = []
    
    for section in measured_positions.keys():
        if section not in model_pos_iso or section not in model_pos_rt:
            continue
            
        for i, pos in enumerate(measured_positions[section]):
            meas_pos = np.array(pos)
            
            # Get model predictions
            pred_pos_iso = model_pos_iso[section][i].detach().numpy()
            pred_pos_rt = model_pos_rt[section][i].detach().numpy()
            
            # Compute residuals
            res_iso = np.linalg.norm(meas_pos - pred_pos_iso)
            res_rt = np.linalg.norm(meas_pos - pred_pos_rt)
            
            residuals_iso.append(res_iso)
            residuals_rt.append(res_rt)
    
    # Create histogram of residuals
    bins = np.linspace(0, max(max(residuals_iso), max(residuals_rt)), 20)
    plt.hist([residuals_iso, residuals_rt], bins=bins, 
            label=['Isotropic', 'Radial/Tangential'],
            color=['red', 'green'],rwidth=0.85)  

    plt.title("Residual Distributions")
    plt.xlabel("Residual Magnitude (mm)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("graphs/residual_comparison.png")
    
    # --- QUANTITATIVE METRICS ---
    # Calculate log-likelihoods
    ll_iso = -model_iso.negative_log_likelihood(data, hole_indices).item()
    ll_rt = -model_rt.negative_log_likelihood(data, hole_indices).item()
    
    # Count parameters
    k_iso = 3 + 3 * len(model_iso.section_ids)  # r, N, sigma, and 3 params per section
    k_rt = 4 + 3 * len(model_rt.section_ids)    # r, N, sigma_r, sigma_t, and 3 params per section
    
    # Sample size
    n = sum(len(positions) for positions in measured_positions.values())
    
    # Calculate AIC and BIC
    aic_iso = 2 * k_iso - 2 * ll_iso
    aic_rt = 2 * k_rt - 2 * ll_rt
    
    bic_iso = k_iso * np.log(n) - 2 * ll_iso
    bic_rt = k_rt * np.log(n) - 2 * ll_rt
    
    # Likelihood ratio test
    lr_stat = 2 * (ll_rt - ll_iso)
    df = k_rt - k_iso
    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    
    # Print results
    print("\n--- Model Comparison Metrics ---")
    print(f"Isotropic model - Log-likelihood: {ll_iso:.2f}, AIC: {aic_iso:.2f}, BIC: {bic_iso:.2f}")
    print(f"Radial/Tang model - Log-likelihood: {ll_rt:.2f}, AIC: {aic_rt:.2f}, BIC: {bic_rt:.2f}")
    print(f"Likelihood ratio test: χ² = {lr_stat:.2f}, df = {df}, p = {p_value:.4f}")
    print(f"Sigma (Isotropic): {params_iso['sigma']:.4f}")
    print(f"Sigma radial: {params_rt['sigma_r']:.4f}, Sigma tangential: {params_rt['sigma_t']:.4f}")
    
    # --- VISUALIZATION 3: Model comparison metrics ---
    plt.figure(figsize=(10, 6))
    
    # Better scale by plotting the negative values for AIC/BIC
    metrics = ['Log-Likelihood', '-AIC', '-BIC']
    iso_values = [ll_iso, -aic_iso, -bic_iso]
    rt_values = [ll_rt, -aic_rt, -bic_rt]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, iso_values, width, label='Isotropic', color='red', alpha=0.7)
    plt.bar(x + width/2, rt_values, width, label='Radial/Tangential', color='green', alpha=0.7)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Model Comparison Metrics (Higher is Better)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, v in enumerate(iso_values):
        plt.text(i - width/2, v + 5, f"{v:.1f}", ha='center')
    
    for i, v in enumerate(rt_values):
        plt.text(i + width/2, v + 5, f"{v:.1f}", ha='center')
    
    plt.savefig("graphs/metrics_comparison.png")
    plt.show()

if __name__ == "__main__":
    compare_models()