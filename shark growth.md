```python
# Program      : Euler's method modified for shark growth


import numpy as np
import matplotlib.pyplot as plt

print("Model for shark growth")

# we choose the parameters based on the final weight of a Great white shark

# b = 0.3 means the shark growth stabilises after 35-40 years

                                        # parameter a
b = 0.3                                 # parameter b
M_max = 2000  # we assume the max weight of our shark is 2000 kg which is realistic for a great white shark

# we calculate alpha based on b and the final weight of the shark. Set dm/dt = 0 and solve for alpha
a = b * (M_max)**(1/3) 
#print(a)

Dt = 0.1                                # timestep Delta t
M_init = 5                              # initial Mass (M)
t_init = 0                              # initial time
t_end = 50                              # stopping time (increased to see the curve evolve)
n_steps = int(round((t_end-t_init)/Dt)) # total number of timesteps

# --- INITIALIZATION ---

Dt = 0.01                               # timestep (0.01 years)
t_end = 80                              # End of simulation (Lifespan)
n_steps = int(round(t_end/Dt))          # total timesteps

t_arr = np.zeros(n_steps + 1)           
M_arr = np.zeros(n_steps + 1)           

t_arr[0] = 0                            # Birth time
M_arr[0] = 25                           # Birth mass (kg)


# Euler's method

for i in range (1, n_steps + 1):
    M = M_arr[i-1]
    t = t_arr[i-1]
    
    # The differential equation:
    dMdt = a * M**(2/3) - b * M         
    
    M_arr[i] = M + Dt*dMdt              
    t_arr[i] = t + Dt                   

# Plot the results

fig = plt.figure(figsize=(10,6))                   
plt.plot(t_arr, M_arr, linewidth = 4)   

plt.title('Great White Shark Growth', fontsize = 25)  
plt.xlabel('age (years)', fontsize = 20)
plt.ylabel('Mass (kg)', fontsize = 20)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.grid(True)    

plt.axhline(y=M_max, color='r', linestyle='--', label='Max Theoretical Size')
plt.legend()

plt.show()                              
fig.savefig('Simulation_Result.jpg', dpi=fig.dpi, bbox_inches = "tight")
```


```python
import numpy as np
import matplotlib.pyplot as plt

# --- PARAMETERS ---
b = 0.3
M_max = 2000
a = b * (M_max)**(1/3)

# Define plot ranges
t_end = 80 # this is approximately lifespan of the great white shark
M_plot_max = 3000  #

# --- GRID GENERATION ---
# We use a normalized 0-1 grid so that the arrows look nice
n = 20
x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))

# Calculate slope at every point
M = y * M_plot_max          # Convert normalized grid back to real Mass
dMdt = a * M**(2/3) - b * M 

# Convert slope to visual units (0-1 scale)
slope_visual = dMdt * (t_end / M_plot_max)

# Create arrow vectors
mag = np.sqrt(1 + slope_visual**2)
u, v = 1/mag, slope_visual/mag

# --- CALCULATE CURVE ---
Dt = 0.1
t_steps = np.arange(0, t_end + Dt, Dt)
M_sol = [25] # Initial mass

for _ in range(len(t_steps)-1):
    M = M_sol[-1]
    change = a * M**(2/3) - b * M
    M_sol.append(M + change * Dt)

# --- PLOTTING ---
fig = plt.figure(figsize=(10, 6))

# 1. Plot Arrows
plt.quiver(x, y, u, v, color='gray', pivot='mid', scale=30, headaxislength=4, width=0.003)

# 2. Plot Solution Curve (Normalized)
plt.plot(t_steps/t_end, np.array(M_sol)/M_plot_max, linewidth=4, label='Shark Growth')

# 3. Plot Asymptote
plt.axhline(M_max/M_plot_max, color='r', linestyle='--', label='Max Theoretical Size')

# 4. Fix Labels (Fake the axes to show real numbers)
plt.xticks(np.linspace(0, 1, 9), np.linspace(0, t_end, 9).astype(int), fontsize=15)
plt.yticks(np.linspace(0, 1, 7), np.linspace(0, M_plot_max, 7).astype(int), fontsize=15)

plt.title('Direction Field: Shark Growth', fontsize=25)
plt.xlabel('age (years)', fontsize=20)
plt.ylabel('Mass (kg)', fontsize=20)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower right')

plt.show()
fig.savefig('Direction_Field.jpg', dpi=fig.dpi, bbox_inches="tight")
```


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. Define the Generalized VBGM Differential Equation ---
# dW/dt = eta * W^p - kappa * W
def growth_model(W, t, eta, kappa, p):
    # Prevent negative mass issues in solver
    if W < 0: return 0
    return eta * W**p - kappa * W

# --- 2. Baseline Parameters (Blue Shark Female based on PDF) ---
# L_inf = 304 cm -> converted to approx W_inf = 337 kg
# k = 0.104
W_inf_base = 337
k_base = 0.104
p_base = 2/3  # Standard Von Bertalanffy

# Derived energetic parameters
kappa_base = 3 * k_base
eta_base = kappa_base * (W_inf_base**(1 - p_base))

# Time settings
t = np.linspace(0, 50, 200)
W0 = 1.0 # Pup biomass (approx)

# --- 3. Scenarios ---
scenarios = [
    {"label": "Baseline (Blue Shark)", "eta": eta_base, "kappa": kappa_base},
    {"label": "High Metabolism (Warmer Ocean)", "eta": eta_base, "kappa": kappa_base * 1.5},
    {"label": "High Intake (Aggressive Hunter)", "eta": eta_base * 1.3, "kappa": kappa_base},
    {"label": "Starvation (Low Intake)", "eta": eta_base * 0.8, "kappa": kappa_base}
]

# --- 4. Plotting ---
plt.figure(figsize=(12, 7))

for scen in scenarios:
    # Solve ODE
    w_sol = odeint(growth_model, W0, t, args=(scen["eta"], scen["kappa"], p_base))
    
    # Calculate resultant asymptotic weight for legend
    # W_inf = (eta / kappa)^(1/(1-p))
    w_inf_calc = (scen["eta"] / scen["kappa"])**(1/(1-p_base))
    
    plt.plot(t, w_sol, linewidth=2.5, label=f"{scen['label']} ($W_\infty \\approx {w_inf_calc:.0f}$ kg)")

plt.title(f"Parameter Sensitivity: $\eta$ (Anabolism) vs $\kappa$ (Catabolism)", fontsize=16)
plt.xlabel("Age (Years)", fontsize=14)
plt.ylabel("Weight (kg)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```


```python
# Program      : Euler's method modified for shark growth (Blind Test)

import numpy as np
import matplotlib.pyplot as plt

print("Model for shark growth - Numerical Verification")

# --- 1. BLIND PARAMETERS ---
# We pick arbitrary coefficients for a and b 
# We do Not define M_max (this prevents circular reasoning
a = 3.8   # Anabolism coefficient (growth rate)
b = 0.3   # Catabolism coefficient (energy loss rate)

# --- 2. ANALYTICAL PREDICTION ---
# We calculate the theoretical limit using the formula: M = (a/b)^3
M_predicted = (a / b)**3 

print(f"Inputs: a={a}, b={b}")
print(f"Mathematical Theory predicts equilibrium at: {M_predicted:.2f} kg")

# --- 3. SIMULATION SETUP ---
Dt = 0.05                               # timestep (0.05 years)
t_end = 80                              # End of simulation (Lifespan)
n_steps = int(round(t_end/Dt))          # total timesteps

t_arr = np.zeros(n_steps + 1)           
M_arr = np.zeros(n_steps + 1)           

t_arr[0] = 0                            # Birth time
M_arr[0] = 25                           # Birth mass (kg)

# --- 4. EULER'S METHOD ---
for i in range (1, n_steps + 1):
    M = M_arr[i-1]
    t = t_arr[i-1]
    
    # The differential equation:
    dMdt = a * M**(2/3) - b * M         
    
    M_arr[i] = M + Dt*dMdt              
    t_arr[i] = t + Dt                   

print(f"Simulation finished at: {M_arr[-1]:.2f} kg")

# --- 5. PLOT THE RESULTS ---
fig = plt.figure(figsize=(10,6))                   

# Plot the simulation line
plt.plot(t_arr, M_arr, linewidth = 4, label='Numerical Simulation')   

# Plot the predicted math value to verify they match
plt.axhline(y=M_predicted, color='r', linestyle='--', label=f'Predicted Equilibrium ({int(M_predicted)} kg)')

plt.title('Verification, fontsize = 20)  
plt.xlabel('Age (years)', fontsize = 15)
plt.ylabel('Mass (kg)', fontsize = 15)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True, alpha=0.3)    
plt.legend(fontsize=12)

plt.show()                              
fig.savefig('Blind_Test_Result.jpg', dpi=fig.dpi, bbox_inches = "tight")
```


```python

```
