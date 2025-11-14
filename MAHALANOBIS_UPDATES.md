# Mahalanobis Distance Implementation for Latent Space Conformal Predictions

## Summary

Updated all conformal prediction scripts in `UQ/conformal/` and `UQ/plotting_paper.py` to support Mahalanobis distance-based metrics for evaluating latent space residuals, matching the implementation in mphys-surrogate-model.

## Files Modified

### 1. **UQ/conformal/ae_SINDy.py**
- **Added imports**: `from sklearn.covariance import LedoitWolf` and `from joblib import Parallel, delayed`
- **Added CLI argument**: `-n/--n_jobs` for parallel computation of Mahalanobis metrics
- **Added function**: `compute_scores_and_inv(res_t)` - Computes Mahalanobis distance scores and inverse covariance matrix using Ledoit-Wolf estimation
- **Modified "full" method**: Added Mahalanobis computation for latent space residuals after conformal predictions
- **Modified "split" method**: Added Mahalanobis computation for latent space residuals using calibration data
- **Updated pickle output**: Includes `latent_maha` dictionary containing:
  - `Sigma_inv`: Inverse covariance matrix at each time step, shape (n_time, latent_dim, latent_dim)
  - `mu`: Mean of residuals at each time step, shape (n_time, latent_dim)
  - `taus`: Quantiles of Mahalanobis scores per alpha value

### 2. **UQ/conformal/ae_SINDy_cv.py**
- **Added imports**: `from sklearn.covariance import LedoitWolf` and `from joblib import Parallel, delayed`
- **Added CLI argument**: `-n/--n_jobs` for parallel Mahalanobis computation
- **Added function**: `compute_scores_and_inv(res_t)` - Same implementation as ae_SINDy.py
- **Modified fold aggregation**: Added Mahalanobis computation on pooled residuals from all CV folds for latent space
- **Updated pickle output**: Includes `latent_maha_cv` dictionary with same structure as ae_SINDy.py

### 3. **UQ/conformal/cp_test.py**
- **Updated pickle loading**: Now supports backward compatibility with old format (5 elements) and new format with Mahalanobis (6 elements)
- **Modified latent space coverage testing**: When `latent_maha` is present:
  - Uses ellipsoidal (Mahalanobis) distance-based coverage evaluation instead of axis-aligned bounds
  - Computes Mahalanobis distance at each time step
  - Compares against quantile thresholds (`taus`)

### 4. **UQ/plotting_paper.py**
- **Updated pickle loading**: Same backward compatibility as cp_test.py
- Prepares infrastructure for Mahalanobis-based visualization (future enhancement)

## Technical Details

### Mahalanobis Distance Computation

For each time step `t` in the latent space:

1. **Residuals**: `res_t = DSD_pred[i] - z_encoded_actual`, shape (n_samples, latent_dim)
2. **Mean centering**: `res_c = res_t - mean(res_t, axis=0)`
3. **Covariance estimation**: Use Ledoit-Wolf shrinkage estimator on `res_c`
4. **Inverse computation**: `Sigma_inv = lw.precision_` (inverse covariance matrix)
5. **Mahalanobis scores**: `score = sqrt(sum((res_c @ Sigma_inv) * res_c, axis=1))`
6. **Quantile thresholds**: For each alpha, compute `tau[alpha] = quantile(scores, 1-alpha)`

### Pickle File Format

**Old format (backward compatible)**:
```python
[alphas, test_size, idx_test, (lower, upper, rep_DSD), (lower_m, upper_m, rep_m)]
```

**New format**:
```python
[alphas, test_size, idx_test, (lower, upper, rep_DSD), (lower_m, upper_m, rep_m), latent_maha]
```

where `latent_maha` is either `None` (for old data) or a dictionary:
```python
{
    "Sigma_inv": np.ndarray of shape (n_time, latent_dim, latent_dim),
    "mu": np.ndarray of shape (n_time, latent_dim),
    "taus": dict mapping alpha values to quantile arrays of shape (n_time,)
}
```

## Benefits

1. **Better uncertainty quantification**: Ellipsoidal confidence regions capture correlations in latent space
2. **Mahalanobis-based coverage**: More statistically principled than axis-aligned bounds
3. **Ledoit-Wolf estimation**: Robust covariance estimation even with high-dimensional latent spaces
4. **Parallelization**: Mahalanobis computation across time steps can be parallelized with `-n/--n_jobs`
5. **Backward compatibility**: Existing pickle files (without Mahalanobis data) still load and work with axis-aligned bounds

## Usage Examples

### Generate conformal predictions with Mahalanobis metrics
```bash
# For full conformal predictions
python UQ/conformal/ae_SINDy.py <dataset> -m full -n 4

# For split conformal with 30% calibration
python UQ/conformal/ae_SINDy.py <dataset> -m split30 -n 4

# For cv+ with 5 folds
python UQ/conformal/ae_SINDy_cv.py <dataset> -k 5 -n 4
```

### Test coverage using Mahalanobis metrics
```bash
# cp_test.py automatically uses Mahalanobis if present in pickle
python UQ/conformal/cp_test.py <dataset> -s latent -a SINDy -m full
```

### Plot with Mahalanobis-aware visualization
```bash
python UQ/plotting_paper.py <dataset> -s latent -a SINDy -m full -t "0 10 20" --ids 0 1 2
```

## Verification

All files have been syntax-checked with `python3 -m py_compile`:
- `UQ/conformal/ae_SINDy.py`
- `UQ/conformal/ae_SINDy_cv.py`
- `UQ/conformal/cp_test.py`
- `UQ/plotting_paper.py`
