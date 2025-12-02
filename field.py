import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_macro_field_figure():
    # 1. Setup Grid
    x = np.linspace(-10, 110, 200)
    y = np.linspace(-10, 110, 200)
    X, Y = np.meshgrid(x, y)

    # 2. Define Positions
    informed_pos = np.array([80, 80])  # Source of Information
    target_cluster_pos = np.array([30, 30])  # Destination (Targets)

    # 3. Define Virtual Potential Field (V_aug)
    sigma_info = 25
    sigma_target = 20

    Z_info = np.exp(-((X - informed_pos[0]) ** 2 + (Y - informed_pos[1]) ** 2) / (2 * sigma_info ** 2))
    Z_target = -1.5 * np.exp(
        -((X - target_cluster_pos[0]) ** 2 + (Y - target_cluster_pos[1]) ** 2) / (2 * sigma_target ** 2))

    V_aug = Z_info + Z_target  # Total Potential

    # 4. Calculate Gradients (Flux J)
    dy, dx = np.gradient(-V_aug)  # Negative gradient for flow

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # A. Heatmap (Scalar Field)
    cp = ax.contourf(X, Y, V_aug, levels=50, cmap='RdBu_r', alpha=0.6)
    # FIX 1: Use raw string r'' for LaTeX
    cbar = fig.colorbar(cp, ax=ax, label=r'Virtual Information Potential $\mathcal{V}_{aug}$')

    # B. Streamlines (Vector Flux)
    strm = ax.streamplot(X, Y, dx, dy, color='k', linewidth=1, density=1.0, arrowsize=1.5)

    # C. Annotations
    # Informed Herder
    ax.scatter(*informed_pos, color='green', s=300, marker='D', edgecolors='white', zorder=10,
               label='Informed Herder (Source)')
    ax.text(informed_pos[0], informed_pos[1] + 5, 'Information\nInjection', ha='center', color='green',
            fontweight='bold')

    # Targets
    ax.scatter(*target_cluster_pos, color='red', s=300, marker='^', edgecolors='white', zorder=10,
               label='Target Cluster (Sink)')
    # FIX 2: Split string or use raw string to avoid \r (rho) issue
    ax.text(target_cluster_pos[0], target_cluster_pos[1] - 10, 'Target Density\n' + r'$\rho_T$', ha='center',
            color='red', fontweight='bold')

    # Standard Herder Flux Label
    # FIX 3: Use raw string to avoid \b (mathbf) issue
    ax.text(55, 55, r'Herder Flux $\mathbf{J}_{S}$', ha='center', rotation=45, fontsize=12, fontweight='bold',
            color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # D. Confinement Interface
    interface = plt.Circle(target_cluster_pos, 15, color='purple', fill=False, linestyle='--', linewidth=3,
                           label='Confinement Interface')
    ax.add_patch(interface)
    ax.text(target_cluster_pos[0] + 15, target_cluster_pos[1] + 15, 'Interface\n(Flux Balance)', color='purple',
            fontsize=10)

    # Styling
    ax.set_title('Macroscopic Field Emergence: Information-Driven Advection', fontsize=14)
    ax.set_xlabel('Spatial Coordinate x')
    ax.set_ylabel('Spatial Coordinate y')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    plt.tight_layout()
    # Save the figure
    plt.savefig('macro_field_emergence.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_macro_field_figure()