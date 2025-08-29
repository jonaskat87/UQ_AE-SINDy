import torch
import numpy as np
from sklearn.mixture import GaussianMixture


class AdaptiveThresholdAnalyzer:
    """Analyzes coefficient distributions to determine thresholding decisions"""

    def __init__(
        self,
        method="bimodal_gmm",
        min_epochs_between=50,
        convergence_patience=10,
        convergence_threshold=0.05,
    ):
        self.method = method
        self.min_epochs_between = min_epochs_between
        self.convergence_patience = convergence_patience
        self.convergence_threshold = convergence_threshold
        self.last_threshold_epoch = 0
        self.coefficient_history = []
        self.loss_history = []

    def should_threshold(self, coeffs, epoch, loss=None):
        """
        Determine if thresholding should occur based on coefficient distribution
        """
        # Minimum epoch spacing requirement
        if epoch - self.last_threshold_epoch < self.min_epochs_between:
            return False, None, {}

        # Store history for convergence analysis
        self.coefficient_history.append(coeffs.clone())
        if loss is not None:
            self.loss_history.append(loss)

        # Check convergence if we have enough history
        if len(self.coefficient_history) >= self.convergence_patience:
            if self._check_convergence():
                threshold, analysis = self._compute_threshold(coeffs)
                if threshold is not None:
                    self.last_threshold_epoch = epoch
                    return True, threshold, analysis

        return False, None, {}

    def _check_convergence(self):
        """Check if coefficients have converged recently"""
        if len(self.coefficient_history) < self.convergence_patience:
            return False

        recent_coeffs = torch.stack(
            self.coefficient_history[-self.convergence_patience :]
        )

        # Compute coefficient change rate
        coeff_changes = []
        for i in range(1, len(recent_coeffs)):
            change = torch.norm(recent_coeffs[i] - recent_coeffs[i - 1]) / torch.norm(
                recent_coeffs[i - 1] + 1e-8
            )
            coeff_changes.append(change.item())

        avg_change = np.mean(coeff_changes)
        return avg_change < self.convergence_threshold

    def _compute_threshold(self, coeffs):
        """Compute threshold based on coefficient distribution"""
        coeffs_flat = coeffs.flatten().abs().detach().cpu().numpy()
        coeffs_nonzero = coeffs_flat[coeffs_flat > 1e-12]  # Remove true zeros

        if len(coeffs_nonzero) < 10:  # Too few coefficients
            return None, {}

        analysis = {"n_coeffs": len(coeffs_nonzero)}

        if self.method == "bimodal_gmm":
            return self._gmm_threshold(coeffs_nonzero, analysis)
        elif self.method == "knee_detection":
            return self._knee_detection_threshold(coeffs_nonzero, analysis)
        else:
            raise ValueError(f"Unknown thresholding method: {self.method}")

    def _gmm_threshold(self, coeffs, analysis):
        """Use Gaussian Mixture Model to detect bimodal distribution"""
        try:
            log_coeffs = np.log10(coeffs + 1e-12).reshape(-1, 1)

            # Fit 2-component GMM
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(log_coeffs)

            # Check if bimodal (well-separated components)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            separation = abs(means[1] - means[0]) / np.mean(stds)

            analysis.update(
                {
                    "separation": separation,
                    "means": means,
                    "stds": stds,
                    "weights": gmm.weights_,
                }
            )

            if separation > 2.0:  # Well-separated components
                # Threshold at intersection of components or weighted average
                low_mean, high_mean = sorted(means)
                threshold_log = (low_mean + high_mean) / 2
                threshold = 10**threshold_log
                analysis["threshold_method"] = "gmm_intersection"
                return threshold, analysis

        except Exception as e:
            analysis["gmm_error"] = str(e)

        return None, analysis

    def _knee_detection_threshold(self, coeffs, analysis):
        """Detect knee/elbow in sorted coefficient curve"""
        sorted_coeffs = np.sort(coeffs)[::-1]  # Descending order
        n = len(sorted_coeffs)

        if n < 20:  # Need enough points
            return None, analysis

        # Use second derivative to find knee
        log_coeffs = np.log10(sorted_coeffs + 1e-12)

        # Smooth if needed
        if n > 50:
            from scipy.ndimage import gaussian_filter1d

            log_coeffs = gaussian_filter1d(log_coeffs, sigma=1)

        # Compute second derivative
        second_deriv = np.diff(log_coeffs, n=2)

        # Find knee (maximum curvature)
        knee_idx = np.argmax(np.abs(second_deriv)) + 2  # Adjust for diff operations

        if knee_idx < len(sorted_coeffs):
            threshold = sorted_coeffs[knee_idx]
            analysis.update(
                {
                    "knee_idx": knee_idx,
                    "knee_value": threshold,
                    "threshold_method": "knee_detection",
                }
            )
            return threshold, analysis

        return None, analysis


class AdaptiveSequentialThresholdingSINDy:
    def __init__(self, model, analyzer):
        self.model = model
        self.analyzer = analyzer
        self.threshold_history = []

    def maybe_apply_threshold(self, epoch, loss=None):
        """Check if thresholding should be applied and do it if so"""
        coeffs = self.model.get_coeffs()

        should_thresh, threshold, analysis = self.analyzer.should_threshold(
            coeffs, epoch, loss
        )

        if should_thresh and threshold is not None:
            n_active = self._apply_threshold(threshold)

            self.threshold_history.append(
                {
                    "epoch": epoch,
                    "threshold": threshold,
                    "n_active": n_active,
                    "analysis": analysis,
                }
            )

            return True, threshold, n_active, analysis

        return False, None, None, {}

    def _apply_threshold(self, threshold):
        """Apply thresholding with given threshold value"""
        with torch.no_grad():
            coeffs = self.model.get_coeffs()
            new_mask = torch.abs(coeffs) >= threshold
            self.model.update_mask(new_mask)
            return torch.sum(self.model.mask).item()
