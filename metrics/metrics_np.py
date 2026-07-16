"""
A1-A12 + A15 metrics — thin wrapper over the reference tsg-benchmark metrics.py.
All metric functions are imported directly from the reference repo to avoid duplication.

Reference: /home/tbasseras/tsg-benchmark/metrics.py
"""
import sys
sys.path.insert(0, "/home/tbasseras/tsg-benchmark")

from metrics import (
    mmd2           as mmd2,            # A1  Full path MMD^2
    terminal_mmd2  as terminal_mmd2,   # A2  Terminal MMD^2
    increment_mmd2 as increment_mmd2,  # A3  Increment MMD^2
    volatility_mmd as volatility_mmd,  # A4  Volatility MMD
    terminal_swd   as terminal_swd,    # A5  Terminal SWD
    path_swd       as path_swd,        # A6  Path SWD
    terminal_cov_error     as terminal_cov_error,    # A7
    terminal_mean_rmse     as terminal_mean_rmse,    # A8
    return_std_error       as return_std_error,      # A9
    return_kurtosis_error  as return_kurtosis_error, # A10
    acf_error              as acf_error,             # A11/A12
    teacher_sigma_metrics  as teacher_sigma_metrics, # A15
)

__all__ = [
    "mmd2", "terminal_mmd2", "increment_mmd2", "volatility_mmd",
    "terminal_swd", "path_swd",
    "terminal_cov_error", "terminal_mean_rmse",
    "return_std_error", "return_kurtosis_error",
    "acf_error",
    "teacher_sigma_metrics",
]
