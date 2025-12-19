#!/usr/bin/env python3
"""
Generate a new shark growth dataset sorted by region using verified VBGM parameters from Table 3.

This script generates approximately 500 data points for shortfin mako shark growth
using verified von Bertalanffy Growth Model (VBGM) parameters from published studies.

The von Bertalanffy Growth Model (VBGM) equations:
- Standard form with t0: L(t) = L_inf * (1 - exp(-k * (t - t0)))
- Alternative form with L0: L(t) = L_inf - (L_inf - L0) * exp(-k * t)

Length-Weight Conversion: W(kg) = 4.4e-6 * FL(cm)^3.14 (Kohler et al. 1996)

Source: Table 3 - Growth studies conducted with shortfin mako by authors in different locations
        [updated from Natanson et al. (2006)]
        Image: data/Screenshot 2025-12-19 at 10.44.35.png

Usage:
    python scripts/generate_region_dataset.py

Output:
    data/mako_growth_data_by_region.csv
"""

import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

# Region code mapping to ensure unique prefixes for shark IDs
REGION_CODES = {
    "China": "CHN",
    "Pacific_Australia": "PAU",
    "Pacific_Baja": "PBJ",
    "Pacific_California": "PCA",
    "Pacific_Chile": "PCH",
    "Pacific_New_Zealand": "PNZ",
    "Southwest_SA": "SSA",
    "West_and_Central_SA": "WCSA",
    "Western_NA": "WNA",
    "Western_and_Central_NP": "WCNP",
}

# Model parameters from Table 3 (image: Screenshot 2025-12-19 at 10.44.35.png)
# Format: region, reference, sex, fl_min, fl_max, linf, k, t0_or_L0, value_type, n_samples, longevity
PARAMETERS = [
    # China - Hsu (2003)
    {"region": "China", "reference": "Hsu_2003", "sex": "M", "fl_min": 72.6, "fl_max": 250.9, 
     "linf": 321.8, "k": 0.04, "t0": -6.07, "n_samples": 133, "longevity": 3},
    {"region": "China", "reference": "Hsu_2003", "sex": "F", "fl_min": 72.6, "fl_max": 314.9, 
     "linf": 403.62, "k": 0.04, "t0": -5.27, "n_samples": 174, "longevity": None},
    
    # Pacific, Australia - Chan (2001)
    {"region": "Pacific_Australia", "reference": "Chan_2001", "sex": "M", "fl_min": 66, "fl_max": 274, 
     "linf": 267, "k": 0.31, "t0": -0.95, "n_samples": 24, "longevity": 9},
    {"region": "Pacific_Australia", "reference": "Chan_2001", "sex": "F", "fl_min": 74, "fl_max": 314, 
     "linf": 349, "k": 0.15, "t0": -1.97, "n_samples": 52, "longevity": 17},
    
    # Pacific, Baja - Ribot-Carballal et al. (2005)
    {"region": "Pacific_Baja", "reference": "Ribot_Carballal_2005", "sex": "M", "fl_min": 68.6, "fl_max": 264, 
     "linf": 375.4, "k": 0.05, "t0": -4.7, "n_samples": 109, "longevity": 55},
    {"region": "Pacific_Baja", "reference": "Ribot_Carballal_2005", "sex": "F", "fl_min": 68.6, "fl_max": 264, 
     "linf": 375.4, "k": 0.05, "t0": -4.7, "n_samples": 109, "longevity": None},
    
    # Pacific, California - Cailliet and Bedford (1983)
    {"region": "Pacific_California", "reference": "Cailliet_Bedford_1983", "sex": "M", "fl_min": 80.6, "fl_max": 293, 
     "linf": 298, "k": 0.07, "t0": -3.75, "n_samples": 44, "longevity": 38},
    {"region": "Pacific_California", "reference": "Cailliet_Bedford_1983", "sex": "F", "fl_min": 80.6, "fl_max": 293, 
     "linf": 298, "k": 0.07, "t0": -3.75, "n_samples": 44, "longevity": None},
    
    # Pacific, Chile - Cerna and Licandeo (2009)
    {"region": "Pacific_Chile", "reference": "Cerna_Licandeo_2009", "sex": "M", "fl_min": 70, "fl_max": 258, 
     "linf": 268.07, "k": 0.08, "t0": -3.58, "n_samples": 243, "longevity": None},
    {"region": "Pacific_Chile", "reference": "Cerna_Licandeo_2009", "sex": "F", "fl_min": 69, "fl_max": 300, 
     "linf": 295.73, "k": 0.07, "t0": -3.18, "n_samples": 304, "longevity": None},
    
    # Pacific, New Zealand - Bishop et al. (2006)
    {"region": "Pacific_New_Zealand", "reference": "Bishop_2006", "sex": "M", "fl_min": 100, "fl_max": 347, 
     "linf": 302.2, "k": 0.05, "t0": -9.04, "n_samples": 145, "longevity": 48},
    # Note: Female data from New Zealand has unrealistic Linf (820.1), so we skip it
    
    # Southwest SA - Doño et al. (2014)
    {"region": "Southwest_SA", "reference": "Dono_2014", "sex": "M", "fl_min": 81, "fl_max": 250, 
     "linf": 416, "k": 0.03, "t0": -6.18, "n_samples": 116, "longevity": None},
    {"region": "Southwest_SA", "reference": "Dono_2014", "sex": "F", "fl_min": 101, "fl_max": 330, 
     "linf": 580, "k": 0.02, "t0": -7.52, "n_samples": 126, "longevity": None},
    
    # West and Central SA - This study (fit1 - primary fit)
    {"region": "West_and_Central_SA", "reference": "This_study_fit1", "sex": "M", "fl_min": 79, "fl_max": 250, 
     "linf": 328.74, "k": 0.08, "t0": -4.47, "n_samples": 129, "longevity": 23},
    {"region": "West_and_Central_SA", "reference": "This_study_fit1", "sex": "F", "fl_min": 73, "fl_max": 296, 
     "linf": 407.65, "k": 0.04, "t0": -7.08, "n_samples": 109, "longevity": 28},
    
    # Western NA - Pratt and Casey (1983)
    {"region": "Western_NA", "reference": "Pratt_Casey_1983", "sex": "M", "fl_min": 69, "fl_max": 238, 
     "linf": 302, "k": 0.26, "t0": -1, "n_samples": 49, "longevity": 10},
    {"region": "Western_NA", "reference": "Pratt_Casey_1983", "sex": "F", "fl_min": 69, "fl_max": 238, 
     "linf": 345, "k": 0.2, "t0": -1, "n_samples": 54, "longevity": 14},
    
    # Western NA - Natanson et al. (2006) - uses L0 instead of t0
    {"region": "Western_NA", "reference": "Natanson_2006", "sex": "M", "fl_min": 72, "fl_max": 260, 
     "linf": 253.3, "k": 0.12, "L0": 71.6, "n_samples": 118, "longevity": 21},
    {"region": "Western_NA", "reference": "Natanson_2006", "sex": "F", "fl_min": 64, "fl_max": 340, 
     "linf": 365.6, "k": 0.08, "L0": 88.4, "n_samples": 140, "longevity": 38},
    
    # Western and central NP - Semba et al. (2009) - uses L0 instead of t0
    {"region": "Western_and_Central_NP", "reference": "Semba_2009", "sex": "M", "fl_min": 73, "fl_max": 265, 
     "linf": 255, "k": 0.16, "L0": 59.7, "n_samples": 128, "longevity": None},
    {"region": "Western_and_Central_NP", "reference": "Semba_2009", "sex": "F", "fl_min": 73, "fl_max": 330, 
     "linf": 340, "k": 0.09, "L0": 59.7, "n_samples": 147, "longevity": None},
]


def vbgf_t0(t, linf, k, t0):
    """
    Von Bertalanffy Growth Function with t0 parameter.
    
    L(t) = L_inf * (1 - exp(-k * (t - t0)))
    
    Parameters:
        t: Age in years
        linf: Asymptotic length (L_inf) in cm
        k: Growth coefficient (per year)
        t0: Theoretical age at length zero (years)
    
    Returns:
        Fork length in cm
    """
    return linf * (1 - np.exp(-k * (t - t0)))


def vbgf_L0(t, linf, k, L0):
    """
    Von Bertalanffy Growth Function with L0 parameter.
    
    L(t) = L_inf - (L_inf - L0) * exp(-k * t)
    
    Parameters:
        t: Age in years
        linf: Asymptotic length (L_inf) in cm
        k: Growth coefficient (per year)
        L0: Length at birth (cm)
    
    Returns:
        Fork length in cm
    """
    return linf - (linf - L0) * np.exp(-k * t)


def length_to_weight(length_cm):
    """
    Convert fork length (cm) to weight (kg) using Kohler et al. 1996.
    
    W(kg) = 4.4e-6 * FL(cm)^3.14
    
    Parameters:
        length_cm: Fork length in cm
    
    Returns:
        Weight in kg
    """
    return 4.4e-6 * (length_cm ** 3.14)


def estimate_max_age(params):
    """
    Estimate maximum age from longevity data or from model parameters.
    
    If longevity is provided, use it directly.
    Otherwise, estimate from the time to reach 95% of L_inf.
    
    Parameters:
        params: Dictionary of model parameters
    
    Returns:
        Maximum age in years (integer)
    """
    if params.get("longevity") is not None:
        return int(params["longevity"])
    
    # If no longevity data, estimate from time to reach 95% of Linf
    k = params["k"]
    if "t0" in params:
        t0 = params["t0"]
        # Solve for t when L = 0.95 * Linf
        # 0.95 = 1 - exp(-k*(t-t0))
        # exp(-k*(t-t0)) = 0.05
        # t = t0 - ln(0.05)/k
        max_age = t0 - np.log(0.05) / k
    else:
        # With L0, estimate similarly
        max_age = -np.log(0.05) / k
    
    return max(int(max_age), 25)  # Minimum 25 years


def generate_data_for_params(params, target_count):
    """
    Generate data points for a single set of model parameters.
    
    Parameters:
        params: Dictionary of model parameters
        target_count: Number of data points to generate
    
    Returns:
        List of dictionaries, each representing a data point
    """
    data = []
    
    linf = params["linf"]
    k = params["k"]
    region = params["region"]
    reference = params["reference"]
    sex = params["sex"]
    fl_min = params["fl_min"]
    fl_max = params["fl_max"]
    
    # Determine if using t0 or L0
    uses_L0 = "L0" in params
    
    max_age = estimate_max_age(params)
    
    # Generate ages - spread across the lifespan with some randomness
    # Use beta distribution biased toward younger ages (more common in samples)
    ages = np.random.beta(1.5, 2.5, target_count) * max_age
    
    for i, age in enumerate(ages):
        # Calculate length using appropriate VBGF form
        if uses_L0:
            length = vbgf_L0(age, linf, k, params["L0"])
        else:
            length = vbgf_t0(age, linf, k, params["t0"])
        
        # Ensure length is within observed range (with 10% buffer) BEFORE adding noise
        length_clipped = np.clip(length, fl_min * 0.9, fl_max * 1.1)
        
        # Add measurement error (±3% CV typically observed in field studies)
        # Applied AFTER clipping to ensure variation even at asymptotic limits
        noise_factor = 1 + np.random.normal(0, 0.03)
        length_with_noise = length_clipped * noise_factor
        
        # Re-clip to ensure we don't exceed bounds by too much
        length_with_noise = np.clip(length_with_noise, fl_min * 0.85, fl_max * 1.15)
        
        # Calculate weight from noisy length
        weight = length_to_weight(length_with_noise)
        
        # Round age to nearest 0.5 (typical for vertebral band counting)
        age_rounded = round(age * 2) / 2
        
        # Generate shark ID (will be reassigned after sorting)
        region_code = REGION_CODES.get(region, region[:3].upper())
        shark_id = f"{region_code}_{sex}{i+1:03d}"
        
        data.append({
            "shark_id": shark_id,
            "age_years": age_rounded,
            "length_FL_cm": round(length_with_noise, 1),
            "mass_kg": round(weight, 1),
            "sex": sex,
            "region": region,
            "source_study": reference,
            "L_inf_cm": linf,
            "K_per_year": k,
            "t0_or_L0": params.get("L0", params.get("t0")),
            "value_type": "L0" if uses_L0 else "t0"
        })
    
    return data


def generate_dataset(target_total=500):
    """
    Generate the complete dataset with approximately target_total data points.
    
    Parameters:
        target_total: Target number of data points (default 500)
    
    Returns:
        pandas DataFrame with the generated data
    """
    # Weight by original sample sizes for more realistic distribution
    total_original_samples = sum(p["n_samples"] for p in PARAMETERS)
    
    all_data = []
    
    for params in PARAMETERS:
        # Weight count by original study sample size
        weight = params["n_samples"] / total_original_samples
        target_count = max(10, int(target_total * weight))  # At least 10 samples per set
        
        data = generate_data_for_params(params, target_count)
        all_data.extend(data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by region (alphabetically), then by sex, then by age
    df = df.sort_values(["region", "sex", "age_years"])
    
    # Reset shark IDs after sorting to be sequential within each region/sex
    new_ids = []
    for (region, sex), group in df.groupby(["region", "sex"]):
        region_code = REGION_CODES.get(region, region[:3].upper())
        for i, idx in enumerate(group.index):
            new_ids.append((idx, f"{region_code}_{sex}{i+1:03d}"))
    
    for idx, new_id in new_ids:
        df.loc[idx, "shark_id"] = new_id
    
    # Final sort
    df = df.sort_values(["region", "sex", "age_years"]).reset_index(drop=True)
    
    return df


def save_dataset(df, output_path):
    """
    Save the dataset to a CSV file with descriptive header comments.
    
    Parameters:
        df: pandas DataFrame with the generated data
        output_path: Path to the output CSV file
    """
    header = """# Shortfin Mako Shark (Isurus oxyrinchus) Growth Data - SORTED BY REGION
# =========================================================================
# 
# This CSV contains approximately 500 age-length-weight data points generated from
# VERIFIED von Bertalanffy Growth Model (VBGM) parameters published in peer-reviewed studies.
#
# SOURCE: Table 3 - Growth studies conducted with shortfin mako by authors in different locations
#         [updated from Natanson et al. (2006)]
#         Image: data/Screenshot 2025-12-19 at 10.44.35.png
#
# REGIONS INCLUDED (sorted alphabetically):
# - China (Hsu 2003)
# - Pacific_Australia (Chan 2001)
# - Pacific_Baja (Ribot_Carballal 2005)
# - Pacific_California (Cailliet_Bedford 1983)
# - Pacific_Chile (Cerna_Licandeo 2009)
# - Pacific_New_Zealand (Bishop 2006) - Males only (female parameters unrealistic)
# - Southwest_SA (Dono 2014)
# - West_and_Central_SA (This study - fit1)
# - Western_NA (Pratt_Casey 1983, Natanson 2006)
# - Western_and_Central_NP (Semba 2009)
#
# GROWTH EQUATIONS:
# - Von Bertalanffy with t0: L(t) = L_inf * (1 - exp(-k * (t - t0)))
# - Von Bertalanffy with L0: L(t) = L_inf - (L_inf - L0) * exp(-k * t)
#
# LENGTH-WEIGHT RELATIONSHIP:
# W(kg) = 4.4e-6 * FL(cm)^3.14 (Kohler et al. 1996)
#
# NOTES:
# - Data points include realistic measurement noise (±3% CV)
# - Ages derived from realistic age distributions based on longevity data
# - Female Pacific_New_Zealand excluded due to unrealistic Linf (820.1 cm)
# - Older sharks near asymptotic length may show similar sizes due to growth curve plateau
#
# GENERATION SCRIPT: scripts/generate_region_dataset.py
#
"""
    
    with open(output_path, "w") as f:
        f.write(header)
        df.to_csv(f, index=False)


def main():
    """Main function to generate and save the dataset."""
    print("Generating shark growth dataset by region...")
    print("=" * 60)
    
    # Generate the dataset
    df = generate_dataset(target_total=500)
    
    # Print summary statistics
    print(f"\nTotal data points generated: {len(df)}")
    print(f"\nData points per region:")
    print(df.groupby("region").size())
    print(f"\nData points per sex:")
    print(df.groupby("sex").size())
    
    # Determine output path (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_path = os.path.join(repo_root, "data", "mako_growth_data_by_region.csv")
    
    # Save the dataset
    save_dataset(df, output_path)
    
    print(f"\nDataset saved to: {output_path}")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()
