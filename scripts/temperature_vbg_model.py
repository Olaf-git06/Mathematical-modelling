"""
Temperature-Dependent von Bertalanffy Growth Model for Mako Sharks
===================================================================

This module implements temperature-dependent growth modeling for shortfin mako sharks
using the von Bertalanffy growth (VBG) equation with Q10 temperature scaling.

Two model variants are implemented:
- Model S: Single Q10 for both anabolic and catabolic rates
- Model D: Separate Q10_A (anabolic) and Q10_B (catabolic)

Authors: Olaf Smits, Konstantinos Pantelakis, Teun van den Berg
Institution: TU Delft
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, least_squares
from typing import Tuple, Optional, Dict, Any
import warnings


# =============================================================================
# Constants and Default Parameters
# =============================================================================

# Reference temperature for Q10 scaling (°C)
T_REF = 18.0

# Default length-weight relationship parameters for mako sharks
# W = a * L^b (L in cm, W in kg)
LW_A = 4.4e-6
LW_B = 3.14


# =============================================================================
# Length-Weight Conversion Functions
# =============================================================================

def length_to_mass(length_cm: np.ndarray, a: float = LW_A, b: float = LW_B) -> np.ndarray:
    """
    Convert fork length (cm) to body mass (kg) using allometric relationship.
    
    Parameters
    ----------
    length_cm : array-like
        Fork length in centimeters
    a : float
        Coefficient of length-weight relationship
    b : float
        Exponent of length-weight relationship
        
    Returns
    -------
    mass_kg : ndarray
        Body mass in kilograms
    """
    return a * np.asarray(length_cm) ** b


def mass_to_length(mass_kg: np.ndarray, a: float = LW_A, b: float = LW_B) -> np.ndarray:
    """
    Convert body mass (kg) to fork length (cm) using allometric relationship.
    
    Parameters
    ----------
    mass_kg : array-like
        Body mass in kilograms
    a : float
        Coefficient of length-weight relationship
    b : float
        Exponent of length-weight relationship
        
    Returns
    -------
    length_cm : ndarray
        Fork length in centimeters
    """
    return (np.asarray(mass_kg) / a) ** (1.0 / b)


# =============================================================================
# Q10 Temperature Scaling Functions
# =============================================================================

def q10_scale(T: float, T_ref: float, Q10: float) -> float:
    """
    Calculate Q10 temperature scaling factor.
    
    The Q10 coefficient describes how a biological rate changes with a 10°C
    temperature increase. This function returns the multiplicative factor
    for the rate at temperature T relative to T_ref.
    
    Parameters
    ----------
    T : float
        Current temperature (°C)
    T_ref : float
        Reference temperature (°C)
    Q10 : float
        Q10 coefficient (typically 1.5-3.0 for biological processes)
        
    Returns
    -------
    scale_factor : float
        Multiplicative scaling factor for the rate
    """
    return Q10 ** ((T - T_ref) / 10.0)


def arrhenius_scale(T: float, T_ref: float, E_a: float, k_B: float = 8.617e-5) -> float:
    """
    Calculate Arrhenius temperature scaling factor.
    
    Alternative to Q10 scaling based on activation energy.
    
    Parameters
    ----------
    T : float
        Current temperature (°C)
    T_ref : float
        Reference temperature (°C)
    E_a : float
        Activation energy (eV)
    k_B : float
        Boltzmann constant (eV/K), default is 8.617e-5
        
    Returns
    -------
    scale_factor : float
        Multiplicative scaling factor for the rate
    """
    T_K = T + 273.15
    T_ref_K = T_ref + 273.15
    return np.exp(-E_a / k_B * (1.0 / T_K - 1.0 / T_ref_K))


# =============================================================================
# VBG Differential Equation Functions
# =============================================================================

def vbg_ode(t: float, w: np.ndarray, eta: float, kappa: float) -> np.ndarray:
    """
    Von Bertalanffy growth ODE (temperature-independent).
    
    dw/dt = eta * w^(2/3) - kappa * w
    
    Parameters
    ----------
    t : float
        Time (years)
    w : array-like
        Body mass (kg)
    eta : float
        Anabolic coefficient
    kappa : float
        Catabolic coefficient
        
    Returns
    -------
    dwdt : ndarray
        Rate of mass change (kg/year)
    """
    w = np.atleast_1d(w)
    w_safe = np.maximum(w, 1e-10)  # Prevent numerical issues
    return eta * w_safe ** (2.0 / 3.0) - kappa * w_safe


def vbg_ode_temp_single(t: float, w: np.ndarray, eta0: float, kappa0: float,
                        Q10: float, T: float, T_ref: float = T_REF) -> np.ndarray:
    """
    Temperature-dependent VBG ODE - Model S (single Q10).
    
    Both anabolic and catabolic rates scale with the same Q10.
    
    dw/dt = eta(T) * w^(2/3) - kappa(T) * w
    
    where:
        eta(T) = eta0 * Q10^((T - T_ref)/10)
        kappa(T) = kappa0 * Q10^((T - T_ref)/10)
    
    Parameters
    ----------
    t : float
        Time (years)
    w : array-like
        Body mass (kg)
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10 : float
        Common Q10 coefficient for both processes
    T : float
        Temperature (°C)
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    dwdt : ndarray
        Rate of mass change (kg/year)
    """
    w = np.atleast_1d(w)
    w_safe = np.maximum(w, 1e-10)
    
    scale = q10_scale(T, T_ref, Q10)
    eta_T = eta0 * scale
    kappa_T = kappa0 * scale
    
    return eta_T * w_safe ** (2.0 / 3.0) - kappa_T * w_safe


def vbg_ode_temp_dual(t: float, w: np.ndarray, eta0: float, kappa0: float,
                      Q10_A: float, Q10_B: float, T: float, 
                      T_ref: float = T_REF) -> np.ndarray:
    """
    Temperature-dependent VBG ODE - Model D (dual Q10).
    
    Anabolic and catabolic rates scale with different Q10 values.
    
    dw/dt = eta(T) * w^(2/3) - kappa(T) * w
    
    where:
        eta(T) = eta0 * Q10_A^((T - T_ref)/10)
        kappa(T) = kappa0 * Q10_B^((T - T_ref)/10)
    
    Parameters
    ----------
    t : float
        Time (years)
    w : array-like
        Body mass (kg)
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10_A : float
        Q10 coefficient for anabolism
    Q10_B : float
        Q10 coefficient for catabolism
    T : float
        Temperature (°C)
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    dwdt : ndarray
        Rate of mass change (kg/year)
    """
    w = np.atleast_1d(w)
    w_safe = np.maximum(w, 1e-10)
    
    eta_T = eta0 * q10_scale(T, T_ref, Q10_A)
    kappa_T = kappa0 * q10_scale(T, T_ref, Q10_B)
    
    return eta_T * w_safe ** (2.0 / 3.0) - kappa_T * w_safe


# =============================================================================
# Analytical Solutions
# =============================================================================

def vbg_analytical(t: np.ndarray, eta: float, kappa: float, w0: float) -> np.ndarray:
    """
    Analytical solution of the VBG differential equation.
    
    w(t) = [eta/kappa + (w0^(1/3) - eta/kappa) * exp(-kappa*t/3)]^3
    
    Parameters
    ----------
    t : array-like
        Time points (years)
    eta : float
        Anabolic coefficient
    kappa : float
        Catabolic coefficient
    w0 : float
        Initial mass (kg)
        
    Returns
    -------
    w : ndarray
        Mass at each time point (kg)
    """
    t = np.asarray(t)
    w_inf_cbrt = eta / kappa  # w_inf^(1/3)
    w0_cbrt = w0 ** (1.0 / 3.0)
    
    return (w_inf_cbrt + (w0_cbrt - w_inf_cbrt) * np.exp(-kappa * t / 3.0)) ** 3


def asymptotic_mass(eta: float, kappa: float) -> float:
    """
    Calculate asymptotic (equilibrium) mass.
    
    w* = (eta / kappa)^3
    
    Parameters
    ----------
    eta : float
        Anabolic coefficient
    kappa : float
        Catabolic coefficient
        
    Returns
    -------
    w_star : float
        Asymptotic mass (kg)
    """
    return (eta / kappa) ** 3


def asymptotic_mass_temp(eta0: float, kappa0: float, T: float, 
                         Q10_A: float, Q10_B: float, T_ref: float = T_REF) -> float:
    """
    Calculate temperature-dependent asymptotic mass.
    
    Parameters
    ----------
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    T : float
        Temperature (°C)
    Q10_A : float
        Q10 for anabolism
    Q10_B : float
        Q10 for catabolism
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w_star : float
        Temperature-dependent asymptotic mass (kg)
    """
    eta_T = eta0 * q10_scale(T, T_ref, Q10_A)
    kappa_T = kappa0 * q10_scale(T, T_ref, Q10_B)
    return (eta_T / kappa_T) ** 3


# =============================================================================
# Numerical Solution Functions
# =============================================================================

def simulate_growth_model_S(w0: float, t_span: Tuple[float, float], 
                            t_eval: np.ndarray, T: float,
                            eta0: float, kappa0: float, Q10: float,
                            T_ref: float = T_REF) -> np.ndarray:
    """
    Simulate growth trajectory using Model S (single Q10).
    
    Parameters
    ----------
    w0 : float
        Initial mass (kg)
    t_span : tuple
        (t_start, t_end) time span (years)
    t_eval : array-like
        Time points at which to evaluate solution
    T : float
        Temperature (°C)
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10 : float
        Q10 coefficient
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w : ndarray
        Mass at each evaluation time point (kg)
    """
    sol = solve_ivp(
        fun=lambda t, w: vbg_ode_temp_single(t, w, eta0, kappa0, Q10, T, T_ref),
        t_span=t_span,
        y0=[w0],
        t_eval=t_eval,
        method='RK45',
        dense_output=False
    )
    
    if not sol.success:
        warnings.warn(f"ODE solver failed: {sol.message}")
        
    return sol.y[0]


def simulate_growth_model_D(w0: float, t_span: Tuple[float, float],
                            t_eval: np.ndarray, T: float,
                            eta0: float, kappa0: float, 
                            Q10_A: float, Q10_B: float,
                            T_ref: float = T_REF) -> np.ndarray:
    """
    Simulate growth trajectory using Model D (dual Q10).
    
    Parameters
    ----------
    w0 : float
        Initial mass (kg)
    t_span : tuple
        (t_start, t_end) time span (years)
    t_eval : array-like
        Time points at which to evaluate solution
    T : float
        Temperature (°C)
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10_A : float
        Q10 coefficient for anabolism
    Q10_B : float
        Q10 coefficient for catabolism
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w : ndarray
        Mass at each evaluation time point (kg)
    """
    sol = solve_ivp(
        fun=lambda t, w: vbg_ode_temp_dual(t, w, eta0, kappa0, Q10_A, Q10_B, T, T_ref),
        t_span=t_span,
        y0=[w0],
        t_eval=t_eval,
        method='RK45',
        dense_output=False
    )
    
    if not sol.success:
        warnings.warn(f"ODE solver failed: {sol.message}")
        
    return sol.y[0]


def simulate_growth_variable_temp(w0: float, times: np.ndarray, 
                                  temperatures: np.ndarray,
                                  eta0: float, kappa0: float,
                                  Q10_A: float, Q10_B: float,
                                  T_ref: float = T_REF) -> np.ndarray:
    """
    Simulate growth with time-varying temperature (Euler method).
    
    Parameters
    ----------
    w0 : float
        Initial mass (kg)
    times : array-like
        Time points (years)
    temperatures : array-like
        Temperature at each time point (°C)
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10_A : float
        Q10 for anabolism
    Q10_B : float
        Q10 for catabolism
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w : ndarray
        Mass trajectory (kg)
    """
    times = np.asarray(times)
    temperatures = np.asarray(temperatures)
    
    w = np.zeros(len(times))
    w[0] = w0
    
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        T = temperatures[i]
        
        eta_T = eta0 * q10_scale(T, T_ref, Q10_A)
        kappa_T = kappa0 * q10_scale(T, T_ref, Q10_B)
        
        dwdt = eta_T * w[i - 1] ** (2.0 / 3.0) - kappa_T * w[i - 1]
        w[i] = max(w[i - 1] + dt * dwdt, 0.0)
    
    return w


# =============================================================================
# Model Fitting Functions
# =============================================================================

def predict_mass_at_age_S(params: np.ndarray, data: pd.DataFrame,
                          w0: float = 2.5, T_ref: float = T_REF) -> np.ndarray:
    """
    Predict mass at observed age for all observations using Model S.
    
    Uses age-at-capture data from literature sources. Assumes birth mass w0
    and integrates growth from age 0 to observed age.
    
    Parameters
    ----------
    params : array-like
        [eta0, kappa0, Q10]
    data : DataFrame
        Must contain: age_years, mean_sst_C
    w0 : float
        Birth mass (kg), default 2.5 kg for mako sharks
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w_pred : ndarray
        Predicted mass at observed age (kg)
    """
    eta0, kappa0, Q10 = params
    
    w_pred = np.zeros(len(data))
    
    for idx, (i, row) in enumerate(data.iterrows()):
        age = row['age_years']
        T = row['mean_sst_C']
        
        # Use analytical solution with temperature-adjusted parameters
        eta_T = eta0 * q10_scale(T, T_ref, Q10)
        kappa_T = kappa0 * q10_scale(T, T_ref, Q10)
        
        w_pred[idx] = vbg_analytical(age, eta_T, kappa_T, w0)
    
    return w_pred


def predict_mass_at_age_D(params: np.ndarray, data: pd.DataFrame,
                          w0: float = 2.5, T_ref: float = T_REF) -> np.ndarray:
    """
    Predict mass at observed age for all observations using Model D.
    
    Uses age-at-capture data from literature sources. Assumes birth mass w0
    and integrates growth from age 0 to observed age.
    
    Parameters
    ----------
    params : array-like
        [eta0, kappa0, Q10_A, Q10_B]
    data : DataFrame
        Must contain: age_years, mean_sst_C
    w0 : float
        Birth mass (kg), default 2.5 kg for mako sharks
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    w_pred : ndarray
        Predicted mass at observed age (kg)
    """
    eta0, kappa0, Q10_A, Q10_B = params
    
    w_pred = np.zeros(len(data))
    
    for idx, (i, row) in enumerate(data.iterrows()):
        age = row['age_years']
        T = row['mean_sst_C']
        
        # Use analytical solution with temperature-adjusted parameters
        eta_T = eta0 * q10_scale(T, T_ref, Q10_A)
        kappa_T = kappa0 * q10_scale(T, T_ref, Q10_B)
        
        w_pred[idx] = vbg_analytical(age, eta_T, kappa_T, w0)
    
    return w_pred


def residuals_model_S(params: np.ndarray, data: pd.DataFrame,
                      w0: float = 2.5, T_ref: float = T_REF) -> np.ndarray:
    """Compute residuals for Model S fitting."""
    w_pred = predict_mass_at_age_S(params, data, w0, T_ref)
    w_obs = data['mass_kg'].values
    return w_pred - w_obs


def residuals_model_D(params: np.ndarray, data: pd.DataFrame,
                      w0: float = 2.5, T_ref: float = T_REF) -> np.ndarray:
    """Compute residuals for Model D fitting."""
    w_pred = predict_mass_at_age_D(params, data, w0, T_ref)
    w_obs = data['mass_kg'].values
    return w_pred - w_obs


def fit_model_S(data: pd.DataFrame, x0: Optional[np.ndarray] = None,
                bounds: Optional[Tuple] = None, w0: float = 2.5,
                T_ref: float = T_REF) -> Dict[str, Any]:
    """
    Fit Model S (single Q10) using nonlinear least squares.
    
    Parameters
    ----------
    data : DataFrame
        Growth data with columns: age_years, mass_kg, mean_sst_C
    x0 : array-like, optional
        Initial parameter guess [eta0, kappa0, Q10]
    bounds : tuple, optional
        Parameter bounds ((lower,), (upper,))
    w0 : float
        Birth mass (kg)
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    result : dict
        Dictionary containing fitted parameters and diagnostics
    """
    if x0 is None:
        x0 = np.array([2.5, 0.3, 2.0])
    
    if bounds is None:
        bounds = ([0.1, 0.01, 1.0], [10.0, 1.0, 5.0])
    
    result = least_squares(
        fun=residuals_model_S,
        x0=x0,
        bounds=bounds,
        args=(data, w0, T_ref),
        method='trf'
    )
    
    params = result.x
    residuals = result.fun
    sse = np.sum(residuals ** 2)
    
    w_obs = data['mass_kg'].values
    ss_tot = np.sum((w_obs - np.mean(w_obs)) ** 2)
    r_squared = 1.0 - sse / ss_tot
    
    w_pred = predict_mass_at_age_S(params, data, w0, T_ref)
    
    # Calculate AIC and BIC
    n = len(data)
    k = 3  # number of parameters
    sigma2 = sse / n
    log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    return {
        'params': params,
        'param_names': ['eta0', 'kappa0', 'Q10'],
        'residuals': residuals,
        'sse': sse,
        'r_squared': r_squared,
        'predictions': w_pred,
        'aic': aic,
        'bic': bic,
        'n_params': k,
        'n_obs': n,
        'success': result.success,
        'message': result.message
    }


def fit_model_D(data: pd.DataFrame, x0: Optional[np.ndarray] = None,
                bounds: Optional[Tuple] = None, w0: float = 2.5,
                T_ref: float = T_REF) -> Dict[str, Any]:
    """
    Fit Model D (dual Q10) using nonlinear least squares.
    
    Parameters
    ----------
    data : DataFrame
        Growth data with columns: age_years, mass_kg, mean_sst_C
    x0 : array-like, optional
        Initial parameter guess [eta0, kappa0, Q10_A, Q10_B]
    bounds : tuple, optional
        Parameter bounds ((lower,), (upper,))
    w0 : float
        Birth mass (kg)
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    result : dict
        Dictionary containing fitted parameters and diagnostics
    """
    if x0 is None:
        x0 = np.array([2.5, 0.3, 2.0, 2.5])
    
    if bounds is None:
        bounds = ([0.1, 0.01, 1.0, 1.0], [10.0, 1.0, 5.0, 5.0])
    
    result = least_squares(
        fun=residuals_model_D,
        x0=x0,
        bounds=bounds,
        args=(data, w0, T_ref),
        method='trf'
    )
    
    params = result.x
    residuals = result.fun
    sse = np.sum(residuals ** 2)
    
    w_obs = data['mass_kg'].values
    ss_tot = np.sum((w_obs - np.mean(w_obs)) ** 2)
    r_squared = 1.0 - sse / ss_tot
    
    w_pred = predict_mass_at_age_D(params, data, w0, T_ref)
    
    # Calculate AIC and BIC
    n = len(data)
    k = 4  # number of parameters
    sigma2 = sse / n
    log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    return {
        'params': params,
        'param_names': ['eta0', 'kappa0', 'Q10_A', 'Q10_B'],
        'residuals': residuals,
        'sse': sse,
        'r_squared': r_squared,
        'predictions': w_pred,
        'aic': aic,
        'bic': bic,
        'n_params': k,
        'n_obs': n,
        'success': result.success,
        'message': result.message
    }


# =============================================================================
# Model Comparison Functions
# =============================================================================

def compare_models(result_S: Dict, result_D: Dict) -> Dict[str, Any]:
    """
    Compare Model S and Model D using information criteria.
    
    Parameters
    ----------
    result_S : dict
        Result from fit_model_S()
    result_D : dict
        Result from fit_model_D()
        
    Returns
    -------
    comparison : dict
        Dictionary with comparison metrics
    """
    delta_aic = result_D['aic'] - result_S['aic']
    delta_bic = result_D['bic'] - result_S['bic']
    
    # Likelihood ratio test (approximate)
    # Chi-squared test for nested models
    # Model S is nested in Model D (Q10_A = Q10_B = Q10)
    chi_sq = 2 * (result_S['sse'] - result_D['sse']) / (result_D['sse'] / result_D['n_obs'])
    
    # Determine preferred model
    if delta_aic < -2:
        preferred_aic = 'Model D'
    elif delta_aic > 2:
        preferred_aic = 'Model S'
    else:
        preferred_aic = 'No clear preference'
    
    if delta_bic < -2:
        preferred_bic = 'Model D'
    elif delta_bic > 2:
        preferred_bic = 'Model S'
    else:
        preferred_bic = 'No clear preference'
    
    return {
        'aic_S': result_S['aic'],
        'aic_D': result_D['aic'],
        'bic_S': result_S['bic'],
        'bic_D': result_D['bic'],
        'delta_aic': delta_aic,
        'delta_bic': delta_bic,
        'r_squared_S': result_S['r_squared'],
        'r_squared_D': result_D['r_squared'],
        'sse_S': result_S['sse'],
        'sse_D': result_D['sse'],
        'preferred_by_aic': preferred_aic,
        'preferred_by_bic': preferred_bic,
        'lrt_chi_sq': chi_sq
    }


# =============================================================================
# Sensitivity Analysis Functions
# =============================================================================

def sensitivity_temperature(eta0: float, kappa0: float, Q10_A: float, Q10_B: float,
                            T_range: np.ndarray, T_ref: float = T_REF) -> Dict[str, np.ndarray]:
    """
    Analyze sensitivity of asymptotic mass to temperature.
    
    Parameters
    ----------
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    Q10_A : float
        Q10 for anabolism
    Q10_B : float
        Q10 for catabolism
    T_range : array-like
        Temperature values to evaluate
    T_ref : float
        Reference temperature (°C)
        
    Returns
    -------
    results : dict
        Contains temperatures, w_star values, and relative changes
    """
    T_range = np.asarray(T_range)
    w_star = np.array([
        asymptotic_mass_temp(eta0, kappa0, T, Q10_A, Q10_B, T_ref)
        for T in T_range
    ])
    
    w_star_ref = asymptotic_mass_temp(eta0, kappa0, T_ref, Q10_A, Q10_B, T_ref)
    relative_change = (w_star - w_star_ref) / w_star_ref * 100
    
    return {
        'temperature': T_range,
        'w_star': w_star,
        'w_star_ref': w_star_ref,
        'relative_change_percent': relative_change
    }


def sensitivity_Q10(eta0: float, kappa0: float, T: float,
                    Q10_range: np.ndarray, T_ref: float = T_REF,
                    vary: str = 'both') -> Dict[str, np.ndarray]:
    """
    Analyze sensitivity of asymptotic mass to Q10 values.
    
    Parameters
    ----------
    eta0 : float
        Anabolic coefficient at reference temperature
    kappa0 : float
        Catabolic coefficient at reference temperature
    T : float
        Temperature (°C)
    Q10_range : array-like
        Q10 values to evaluate
    T_ref : float
        Reference temperature (°C)
    vary : str
        Which Q10 to vary: 'A', 'B', or 'both'
        
    Returns
    -------
    results : dict
        Contains Q10 values and corresponding w_star values
    """
    Q10_range = np.asarray(Q10_range)
    
    if vary == 'A':
        w_star = np.array([
            asymptotic_mass_temp(eta0, kappa0, T, Q, 2.5, T_ref)
            for Q in Q10_range
        ])
    elif vary == 'B':
        w_star = np.array([
            asymptotic_mass_temp(eta0, kappa0, T, 2.0, Q, T_ref)
            for Q in Q10_range
        ])
    else:  # both
        w_star = np.array([
            asymptotic_mass_temp(eta0, kappa0, T, Q, Q, T_ref)
            for Q in Q10_range
        ])
    
    return {
        'Q10': Q10_range,
        'w_star': w_star,
        'vary': vary
    }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_growth_data(filepath: str) -> pd.DataFrame:
    """
    Load mako shark growth data from CSV file.
    
    The data file should contain age-at-capture data from literature sources
    with temperature assignments for each region.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    data : DataFrame
        Growth data with required columns
        
    Notes
    -----
    Required columns: shark_id, age_years, mass_kg, mean_sst_C
    Optional columns: length_cm, sex, region, source
    """
    data = pd.read_csv(filepath, comment='#')
    
    # Validate required columns
    required_cols = ['shark_id', 'age_years', 'mass_kg', 'mean_sst_C']
    
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return data


# =============================================================================
# Reporting Functions
# =============================================================================

def print_fit_summary(result: Dict[str, Any], model_name: str = "Model") -> None:
    """Print a summary of model fitting results."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} Fitting Summary")
    print('=' * 60)
    
    print("\nParameter Estimates:")
    for name, value in zip(result['param_names'], result['params']):
        print(f"  {name:12s} = {value:.6f}")
    
    print(f"\nModel Fit Statistics:")
    print(f"  SSE          = {result['sse']:.4f}")
    print(f"  R²           = {result['r_squared']:.4f}")
    print(f"  AIC          = {result['aic']:.2f}")
    print(f"  BIC          = {result['bic']:.2f}")
    print(f"  N parameters = {result['n_params']}")
    print(f"  N observations = {result['n_obs']}")
    
    if not result['success']:
        print(f"\n  Warning: {result['message']}")


def print_comparison_summary(comparison: Dict[str, Any]) -> None:
    """Print a summary of model comparison."""
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    
    print("\nFit Statistics:")
    print(f"  Model S: R² = {comparison['r_squared_S']:.4f}, AIC = {comparison['aic_S']:.2f}, BIC = {comparison['bic_S']:.2f}")
    print(f"  Model D: R² = {comparison['r_squared_D']:.4f}, AIC = {comparison['aic_D']:.2f}, BIC = {comparison['bic_D']:.2f}")
    
    print(f"\nModel Selection:")
    print(f"  ΔAIC (D - S) = {comparison['delta_aic']:.2f}")
    print(f"  ΔBIC (D - S) = {comparison['delta_bic']:.2f}")
    print(f"  Preferred by AIC: {comparison['preferred_by_aic']}")
    print(f"  Preferred by BIC: {comparison['preferred_by_bic']}")


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'mako_growth_data.csv')
    
    print("Temperature-Dependent Von Bertalanffy Growth Model")
    print("Shortfin Mako Shark (Isurus oxyrinchus)")
    print("=" * 60)
    print("\nData Sources:")
    print("  - Rolim et al. (2020): South Atlantic population")
    print("  - Natanson et al. (2006): North Atlantic population")
    print("  - Ribot-Carballal et al. (2005): Eastern Pacific (Baja California)")
    print("  - Cerna & Licandeo (2009): Southeast Pacific (Chile)")
    
    try:
        data = load_growth_data(data_path)
        print(f"\nLoaded {len(data)} observations from published literature")
        
        # Show data summary by region
        if 'region' in data.columns:
            print("\nObservations by region:")
            for region in data['region'].unique():
                n = len(data[data['region'] == region])
                temp = data[data['region'] == region]['mean_sst_C'].iloc[0]
                print(f"  {region}: {n} observations (SST = {temp}°C)")
        
        # Fit Model S
        print("\n" + "-" * 60)
        print("Fitting Model S (single Q10)...")
        result_S = fit_model_S(data)
        print_fit_summary(result_S, "Model S (Single Q10)")
        
        # Fit Model D
        print("\n" + "-" * 60)
        print("Fitting Model D (dual Q10)...")
        result_D = fit_model_D(data)
        print_fit_summary(result_D, "Model D (Dual Q10)")
        
        # Compare models
        comparison = compare_models(result_S, result_D)
        print_comparison_summary(comparison)
        
        # Temperature sensitivity
        print("\n" + "=" * 60)
        print("Temperature Sensitivity Analysis")
        print("=" * 60)
        
        eta0, kappa0 = result_D['params'][0], result_D['params'][1]
        Q10_A, Q10_B = result_D['params'][2], result_D['params'][3]
        
        T_range = np.array([14, 16, 18, 20, 22, 24])
        sens = sensitivity_temperature(eta0, kappa0, Q10_A, Q10_B, T_range)
        
        print("\nAsymptotic mass (w*) at different temperatures:")
        for T, w, change in zip(sens['temperature'], sens['w_star'], sens['relative_change_percent']):
            print(f"  T = {T:5.1f}°C: w* = {w:7.1f} kg ({change:+6.1f}%)")
        
        # Climate scenario projections
        print("\n" + "=" * 60)
        print("Climate Change Impact Projections")
        print("=" * 60)
        T_current = T_REF  # Use module constant for consistency
        w_current = asymptotic_mass_temp(eta0, kappa0, T_current, Q10_A, Q10_B, T_REF)
        
        for delta_T, scenario in [(0, "Current"), (2, "2050 (+2°C)"), (4, "2100 (+4°C)")]:
            T = T_current + delta_T
            w = asymptotic_mass_temp(eta0, kappa0, T, Q10_A, Q10_B, T_REF)
            change = (w - w_current) / w_current * 100
            print(f"  {scenario:18s}: w* = {w:7.1f} kg ({change:+6.1f}%)")
        
    except FileNotFoundError:
        print(f"\nData file not found: {data_path}")
        print("Please ensure the data file exists.")
        
        # Demo with literature-based example
        print("\nRunning demo with literature parameters...")
        eta0, kappa0 = 2.24, 0.30
        Q10_A, Q10_B = 2.0, 2.5
        
        print(f"\nExample: eta0={eta0}, kappa0={kappa0}, Q10_A={Q10_A}, Q10_B={Q10_B}")
        
        T_range = np.array([14, 16, 18, 20, 22, 24])
        sens = sensitivity_temperature(eta0, kappa0, Q10_A, Q10_B, T_range)
        
        print("\nAsymptotic mass (w*) at different temperatures:")
        for T, w, change in zip(sens['temperature'], sens['w_star'], sens['relative_change_percent']):
            print(f"  T = {T:5.1f}°C: w* = {w:7.1f} kg ({change:+6.1f}%)")
