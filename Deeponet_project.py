import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint

torch.manual_seed(42)
np.random.seed(42)

# ============================================
# 1. GÉNÉRATION DES DONNÉES
# ============================================

class HeatControlProblem:
    """
    Problème de contrôle optimal de la chaleur 1D
    Équation: du/dt = d²u/dx² + u(x,t) * contrôle(t)
    Objectif: atteindre une température cible
    """
    def __init__(self, nx=50, nt=100, T=1.0, L=1.0):
        self.nx = nx  # points spatiaux
        self.nt = nt  # points temporels
        self.T = T    # temps final
        self.L = L    # longueur du domaine
        
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt)
        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)
        
    def solve_heat_equation(self, u0, control_func):
        """
        Résout l'équation de la chaleur avec contrôle
        u0: condition initiale (fonction de x)
        control_func: fonction de contrôle (fonction de t)
        """
        u = np.zeros((self.nt, self.nx))
        u[0, :] = u0
        
        # Schéma différences finies
        alpha = self.dt / (self.dx ** 2)
        
        for n in range(self.nt - 1):
            control_val = control_func(self.t[n])
            
            for i in range(1, self.nx - 1):
                # Laplacien + terme de contrôle
                laplacian = (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                u[n+1, i] = u[n, i] + alpha * laplacian + self.dt * control_val * u[n, i]
            
            # Conditions aux bords (Dirichlet)
            u[n+1, 0] = 0
            u[n+1, -1] = 0
            
        return u

def generate_training_data(n_samples=100, nx=50, nt=100):
    """
    Génère des données d'entraînement avec différents contrôles
    """
    problem = HeatControlProblem(nx=nx, nt=nt)
    
    # Stockage
    controls = []      # Fonctions de contrôle
    initial_conditions = []  # Conditions initiales
    solutions = []     # Solutions u(x,t)
    
    for _ in range(n_samples):
        # Génère une condition initiale aléatoire
        freq = np.random.uniform(1, 5)
        amplitude = np.random.uniform(0.5, 2.0)
        u0 = amplitude * np.sin(freq * np.pi * problem.x / problem.L)
        
        # Génère une fonction de contrôle aléatoire
        control_type = np.random.choice(['constant', 'linear', 'sinusoidal'])
        
        if control_type == 'constant':
            c_val = np.random.uniform(-0.5, 0.5)
            control_func = lambda t: c_val
            control_values = np.full(nt, c_val)
            
        elif control_type == 'linear':
            slope = np.random.uniform(-1, 1)
            control_func = lambda t: slope * t
            control_values = slope * problem.t
            
        else:  # sinusoidal
            freq = np.random.uniform(1, 5)
            amplitude = np.random.uniform(0.1, 0.5)
            control_func = lambda t: amplitude * np.sin(2 * np.pi * freq * t)
            control_values = amplitude * np.sin(2 * np.pi * freq * problem.t)
        
        # Résout l'équation
        u_solution = problem.solve_heat_equation(u0, control_func)
        
        controls.append(control_values)
        initial_conditions.append(u0)
        solutions.append(u_solution)
    
    return {
        'controls': np.array(controls),
        'initial_conditions': np.array(initial_conditions),
        'solutions': np.array(solutions),
        'x': problem.x,
        't': problem.t
    }

# ============================================
# 2. ARCHITECTURE DEEPONET
# ============================================

class BranchNet(nn.Module):
    """Branch Network: encode la fonction d'entrée (contrôle)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    """Trunk Network: encode les coordonnées (x, t)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    """
    Deep Operator Network pour le contrôle optimal
    """
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=100, basis_dim=100):
        super().__init__()
        self.branch = BranchNet(branch_input_dim, hidden_dim, basis_dim)
        self.trunk = TrunkNet(trunk_input_dim, hidden_dim, basis_dim)
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, control, coords):
        """
        control: (batch, nt) - fonction de contrôle discrétisée
        coords: (batch, n_points, 2) - coordonnées (x, t)
        """
        branch_out = self.branch(control)  # (batch, basis_dim)
        trunk_out = self.trunk(coords)     # (batch, n_points, basis_dim)
        
        # Produit scalaire
        output = torch.einsum('bi,bpi->bp', branch_out, trunk_out) + self.bias
        return output

# ============================================
# 3. ENTRAÎNEMENT
# ============================================

def train_deeponet(model, data, epochs=1000, lr=0.001, batch_size=32):
    """Entraîne le modèle DeepONet"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Préparation des données
    n_samples = data['controls'].shape[0]
    nt = data['controls'].shape[1]
    nx = data['initial_conditions'].shape[1]
    
    # Création des coordonnées (x, t)
    x_grid, t_grid = np.meshgrid(data['x'], data['t'])
    coords = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1)
    
    # Conversion en tensors
    controls_tensor = torch.FloatTensor(data['controls'])
    coords_tensor = torch.FloatTensor(coords)
    solutions_flat = data['solutions'].reshape(n_samples, -1)
    solutions_tensor = torch.FloatTensor(solutions_flat)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            control_batch = controls_tensor[batch_idx]
            solution_batch = solutions_tensor[batch_idx]
            
            # Forward pass
            coords_batch = coords_tensor.unsqueeze(0).repeat(len(batch_idx), 1, 1)
            pred = model(control_batch, coords_batch)
            
            # Loss
            loss = criterion(pred, solution_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return losses

# ============================================
# 4. VISUALISATION ET ÉVALUATION
# ============================================

def visualize_results(model, data, sample_idx=0):
    """Visualise les résultats pour un échantillon"""
    model.eval()
    
    # Préparation
    control = torch.FloatTensor(data['controls'][sample_idx:sample_idx+1])
    x_grid, t_grid = np.meshgrid(data['x'], data['t'])
    coords = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1)
    coords_tensor = torch.FloatTensor(coords).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        pred = model(control, coords_tensor)
        pred_solution = pred.squeeze().numpy().reshape(len(data['t']), len(data['x']))
    
    # Solution exacte
    exact_solution = data['solutions'][sample_idx]
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Solution exacte
    im1 = axes[0, 0].contourf(data['x'], data['t'], exact_solution, levels=20, cmap='hot')
    axes[0, 0].set_title('Solution Exacte')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Prédiction DeepONET
    im2 = axes[0, 1].contourf(data['x'], data['t'], pred_solution, levels=20, cmap='hot')
    axes[0, 1].set_title('Prédiction DeepONET')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Erreur
    error = np.abs(exact_solution - pred_solution)
    im3 = axes[0, 2].contourf(data['x'], data['t'], error, levels=20, cmap='viridis')
    axes[0, 2].set_title(f'Erreur Absolue (Max: {error.max():.4f})')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Fonction de contrôle
    axes[1, 0].plot(data['t'], data['controls'][sample_idx])
    axes[1, 0].set_title('Fonction de Contrôle')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('u(t)')
    axes[1, 0].grid(True)
    
    # Comparaison à t fixé
    t_idx = len(data['t']) // 2
    axes[1, 1].plot(data['x'], exact_solution[t_idx], 'b-', label='Exact', linewidth=2)
    axes[1, 1].plot(data['x'], pred_solution[t_idx], 'r--', label='DeepONET', linewidth=2)
    axes[1, 1].set_title(f'Profil à t = {data["t"][t_idx]:.2f}')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u(x,t)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Erreur relative
    rel_error = np.abs(exact_solution - pred_solution) / (np.abs(exact_solution) + 1e-10)
    axes[1, 2].hist(rel_error.flatten(), bins=50, edgecolor='black')
    axes[1, 2].set_title('Distribution Erreur Relative')
    axes[1, 2].set_xlabel('Erreur Relative')
    axes[1, 2].set_ylabel('Fréquence')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('deeponet_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Métriques
    mse = np.mean((exact_solution - pred_solution) ** 2)
    mae = np.mean(np.abs(exact_solution - pred_solution))
    rel_l2 = np.linalg.norm(exact_solution - pred_solution) / np.linalg.norm(exact_solution)
    
    print(f"\n{'='*50}")
    print(f"MÉTRIQUES DE PERFORMANCE")
    print(f"{'='*50}")
    print(f"MSE (Mean Squared Error):     {mse:.6f}")
    print(f"MAE (Mean Absolute Error):    {mae:.6f}")
    print(f"Relative L2 Error:            {rel_l2:.6f}")
    print(f"{'='*50}\n")

# ============================================
# 5. EXÉCUTION PRINCIPALE
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("DeepONET pour le Contrôle Optimal de la Chaleur")
    print("="*60)
    
    # 1. Génération des données
    print("\n[1/4] Génération des données d'entraînement...")
    data = generate_training_data(n_samples=200, nx=50, nt=100)
    print(f"✓ {len(data['controls'])} échantillons générés")
    
    # 2. Création du modèle
    print("\n[2/4] Création du modèle DeepONET...")
    model = DeepONet(
        branch_input_dim=100,  # nombre de points temporels pour le contrôle
        trunk_input_dim=2,     # (x, t)
        hidden_dim=100,
        basis_dim=100
    )
    print(f"✓ Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    
    # 3. Entraînement
    print("\n[3/4] Entraînement du modèle...")
    losses = train_deeponet(model, data, epochs=1000, lr=0.001, batch_size=32)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe d\'Apprentissage')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n[4/4] Génération des visualisations...")
    visualize_results(model, data, sample_idx=0)
    
    print("\n✓ Terminé ! Les résultats sont sauvegardés.")
    print("  - deeponet_results.png: Comparaison des résultats")
    print("  - training_loss.png: Courbe d'apprentissage")