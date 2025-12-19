```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    "mako_growth_data_verified.csv",
    comment="#"
)

df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shark_id</th>
      <th>age_years</th>
      <th>length_FL_cm</th>
      <th>mass_kg</th>
      <th>sex</th>
      <th>region</th>
      <th>source_study</th>
      <th>source_pdf</th>
      <th>pdf_reference</th>
      <th>growth_model</th>
      <th>L_inf_cm</th>
      <th>K_per_year</th>
      <th>L0_cm</th>
      <th>n_original_samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WES_M001</td>
      <td>0</td>
      <td>71.6</td>
      <td>2.9</td>
      <td>M</td>
      <td>Western_North_Atlantic</td>
      <td>Natanson_2006_WNA_Male</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>vbgf</td>
      <td>253.3</td>
      <td>0.125</td>
      <td>71.6</td>
      <td>118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WES_M002</td>
      <td>1</td>
      <td>93.0</td>
      <td>6.7</td>
      <td>M</td>
      <td>Western_North_Atlantic</td>
      <td>Natanson_2006_WNA_Male</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>vbgf</td>
      <td>253.3</td>
      <td>0.125</td>
      <td>71.6</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WES_M003</td>
      <td>2</td>
      <td>111.8</td>
      <td>11.9</td>
      <td>M</td>
      <td>Western_North_Atlantic</td>
      <td>Natanson_2006_WNA_Male</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>vbgf</td>
      <td>253.3</td>
      <td>0.125</td>
      <td>71.6</td>
      <td>118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WES_M004</td>
      <td>3</td>
      <td>128.4</td>
      <td>18.4</td>
      <td>M</td>
      <td>Western_North_Atlantic</td>
      <td>Natanson_2006_WNA_Male</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>vbgf</td>
      <td>253.3</td>
      <td>0.125</td>
      <td>71.6</td>
      <td>118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>WES_M005</td>
      <td>4</td>
      <td>143.1</td>
      <td>25.8</td>
      <td>M</td>
      <td>Western_North_Atlantic</td>
      <td>Natanson_2006_WNA_Male</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>Validated_age_and_growth_estimates_for_the_sho...</td>
      <td>vbgf</td>
      <td>253.3</td>
      <td>0.125</td>
      <td>71.6</td>
      <td>118</td>
    </tr>
  </tbody>
</table>
</div>




```python
# --- Extract data (adjust column names if needed) ---
age_data = df["age_years"]      # years
mass_data = df["mass_kg"]    # kg




# Pratt and Casey (1983)
k_vb = 0.26
L_inf = 302

# Length–weight relation (Kohler et al. 1995)
a = 5.243e-6
b = 3.1407

w_inf = a * (L_inf ** b)

kappa = 3 * k_vb
eta = kappa * (w_inf ** (1/3))

Dt = 0.01
t_end = 20
n_steps = int(t_end / Dt)

t_arr = np.zeros(n_steps + 1)
M_arr = np.zeros(n_steps + 1)

t_arr[0] = 0
M_arr[0] = 25

for i in range(1, n_steps + 1):
    M = M_arr[i-1]
    dMdt = eta * M**(2/3) - kappa * M
    M_arr[i] = M + Dt * dMdt
    t_arr[i] = t_arr[i-1] + Dt


# --- PLOTTING ---
plt.figure(figsize=(10, 6))

plt.scatter(age_data, mass_data, color="black", alpha=0.6, label="Verified data")
plt.plot(t_arr, M_arr, linewidth=4, label="Baseline model")

plt.axhline(y=w_inf, linestyle="--", color="red",
            label=f"Asymptotic mass = {w_inf:.1f} kg")

plt.xlabel("Age (years)")
plt.ylabel("Mass (kg)")
plt.title("Baseline growth model vs data")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

```


    
![png](output_1_0.png)
    



```python

```
