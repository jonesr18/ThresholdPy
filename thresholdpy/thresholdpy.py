"""
ThresholdPy: A Python adaptation of ThresholdR for CITE-seq denoising with ScanPy integration

This package provides Gaussian Mixture Model-based denoising for ADT data from CITE-seq experiments,
designed to work seamlessly with ScanPy AnnData objects.

Original R package: https://github.com/MDMotlagh/ThresholdR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union
import warnings
import scanpy as sc
from anndata import AnnData
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdPy:
    """
    A Python implementation of ThresholdR for CITE-seq ADT denoising using Gaussian Mixture Models.
    
    This class identifies noise populations in surface markers across cells and calculates
    upper thresholds of noise components to separate expressing and non-expressing cells.
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 max_iter: int = 100,
                 random_state: int = 42,
                 covariance_type: str = 'full'):
        """
        Initialize ThresholdPy with GMM parameters.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of mixture components (typically 2: noise and signal)
        max_iter : int, default=100
            Maximum number of EM iterations
        random_state : int, default=42
            Random state for reproducibility
        covariance_type : str, default='full'
            Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.thresholds_ = {}
        self.fitted_models_ = {}
        self.fit_stats_ = {}
        
    def _validate_adata(self, adata: AnnData, protein_layer: Optional[str] = None) -> np.ndarray:
        """
        Validate AnnData object and extract protein expression data.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data object containing CITE-seq data
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X
            
        Returns:
        --------
        protein_data : np.ndarray
            Protein expression matrix (cells x proteins)
        """
        if not isinstance(adata, AnnData):
            raise TypeError("Input must be an AnnData object")
            
        if protein_layer is not None:
            if protein_layer not in adata.layers:
                raise ValueError(f"Layer '{protein_layer}' not found in adata.layers")
            protein_data = adata.layers[protein_layer]
        else:
            protein_data = adata.X
            
        if protein_data is None:
            raise ValueError("No protein expression data found")
            
        # Convert sparse to dense if necessary
        if hasattr(protein_data, 'toarray'):
            protein_data = protein_data.toarray()
            
        return protein_data
    
    def _fit_gmm_single_protein(self, 
                               protein_values: np.ndarray, 
                               protein_name: str) -> Tuple[GaussianMixture, Dict]:
        """
        Fit GMM to a single protein's expression values.
        
        Parameters:
        -----------
        protein_values : np.ndarray
            Expression values for one protein
        protein_name : str
            Name of the protein
            
        Returns:
        --------
        gmm : GaussianMixture
            Fitted GMM model
        stats : dict
            Fitting statistics
        """
        # Remove zero/negative values and reshape for sklearn
        valid_values = protein_values[protein_values > 0]
        
        if len(valid_values) < 10:
            logger.warning(f"Insufficient non-zero values for {protein_name}: {len(valid_values)}")
            return None, {"converged": False, "n_valid": len(valid_values)}
        
        # Log transform to make data more Gaussian-like
        log_values = np.log1p(valid_values).reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            random_state=self.random_state,
            covariance_type=self.covariance_type
        )
        
        try:
            gmm.fit(log_values)
            
            # Calculate statistics
            aic = gmm.aic(log_values)
            bic = gmm.bic(log_values)
            log_likelihood = gmm.score(log_values)
            
            stats = {
                "converged": gmm.converged_,
                "n_valid": len(valid_values),
                "aic": aic,
                "bic": bic,
                "log_likelihood": log_likelihood,
                "means": gmm.means_.flatten(),
                "weights": gmm.weights_,
                "covariances": gmm.covariances_
            }
            
            return gmm, stats
            
        except Exception as e:
            logger.error(f"Failed to fit GMM for {protein_name}: {str(e)}")
            return None, {"converged": False, "error": str(e)}
    
    def _calculate_threshold(self, gmm: GaussianMixture, stats: Dict) -> float:
        """
        Calculate the threshold separating noise and signal components.
        
        Parameters:
        -----------
        gmm : GaussianMixture
            Fitted GMM model
        stats : dict
            Fitting statistics
            
        Returns:
        --------
        threshold : float
            Threshold value in original scale
        """
        if not stats["converged"] or gmm is None:
            return np.nan
        
        # Identify noise component (typically the one with lower mean)
        means = stats["means"]
        weights = stats["weights"]
        
        # Assume noise component has lower mean
        noise_idx = np.argmin(means)
        noise_mean = means[noise_idx]
        noise_weight = weights[noise_idx]
        
        # Get covariance for noise component
        if self.covariance_type == 'full':
            noise_std = np.sqrt(stats["covariances"][noise_idx][0, 0])
        elif self.covariance_type == 'diag':
            noise_std = np.sqrt(stats["covariances"][noise_idx][0])
        else:  # spherical or tied
            noise_std = np.sqrt(stats["covariances"][noise_idx])
        
        # Calculate threshold as mean + 2*std of noise component (95% confidence)
        log_threshold = noise_mean + 2 * noise_std
        
        # Convert back to original scale
        threshold = np.expm1(log_threshold)
        
        return threshold
    
    def fit(self, 
            adata: AnnData, 
            protein_layer: Optional[str] = None,
            protein_names: Optional[List[str]] = None) -> 'ThresholdPy':
        """
        Fit GMM models to all proteins and calculate thresholds.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data object containing CITE-seq data
        protein_layer : str, optional
            Layer containing protein expression data
        protein_names : list, optional
            Specific proteins to analyze. If None, analyzes all
            
        Returns:
        --------
        self : ThresholdPy
            Fitted ThresholdPy object
        """
        logger.info("Starting ThresholdPy fitting...")
        
        # Validate and extract data
        protein_data = self._validate_adata(adata, protein_layer)
        
        # Get protein names
        if protein_names is None:
            if protein_layer is not None:
                # Try to get feature names from var_names or use indices
                all_proteins = adata.var_names.tolist()
            else:
                all_proteins = adata.var_names.tolist()
        else:
            all_proteins = protein_names
        
        n_proteins = protein_data.shape[1]
        if len(all_proteins) != n_proteins:
            logger.warning(f"Protein names length ({len(all_proteins)}) doesn't match data ({n_proteins})")
            all_proteins = [f"Protein_{i}" for i in range(n_proteins)]
        
        # Fit GMM for each protein
        for i, protein_name in enumerate(all_proteins):
            logger.info(f"Fitting GMM for {protein_name} ({i+1}/{len(all_proteins)})")
            
            protein_values = protein_data[:, i]
            gmm, stats = self._fit_gmm_single_protein(protein_values, protein_name)
            
            if gmm is not None:
                threshold = self._calculate_threshold(gmm, stats)
                self.thresholds_[protein_name] = threshold
                self.fitted_models_[protein_name] = gmm
                self.fit_stats_[protein_name] = stats
            else:
                logger.warning(f"Failed to fit GMM for {protein_name}")
                self.thresholds_[protein_name] = np.nan
        
        logger.info(f"Completed fitting for {len(self.thresholds_)} proteins")
        return self
    
    def transform(self, 
                  adata: AnnData, 
                  protein_layer: Optional[str] = None,
                  inplace: bool = True,
                  output_layer: str = 'protein_denoised') -> Optional[AnnData]:
        """
        Apply thresholds to denoise protein expression data.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data object containing CITE-seq data
        protein_layer : str, optional
            Layer containing protein expression data
        inplace : bool, default=True
            Whether to modify adata in place
        output_layer : str, default='protein_denoised'
            Name of output layer for denoised data
            
        Returns:
        --------
        adata_denoised : AnnData or None
            Denoised data (if inplace=False)
        """
        if not self.thresholds_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Validate and extract data
        protein_data = self._validate_adata(adata, protein_layer)
        
        # Apply thresholds
        denoised_data = protein_data.copy()
        
        protein_names = list(self.thresholds_.keys())
        for i, protein_name in enumerate(protein_names):
            if i >= protein_data.shape[1]:
                break
                
            threshold = self.thresholds_[protein_name]
            if not np.isnan(threshold):
                # Set values below threshold to 0
                denoised_data[:, i] = np.where(
                    protein_data[:, i] < threshold, 0, protein_data[:, i]
                )
        
        if inplace:
            adata.layers[output_layer] = denoised_data
            return None
        else:
            adata_copy = adata.copy()
            adata_copy.layers[output_layer] = denoised_data
            return adata_copy
    
    def fit_transform(self, 
                     adata: AnnData, 
                     protein_layer: Optional[str] = None,
                     protein_names: Optional[List[str]] = None,
                     inplace: bool = True,
                     output_layer: str = 'protein_denoised') -> Optional[AnnData]:
        """
        Fit GMM models and apply thresholds in one step.
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data object containing CITE-seq data
        protein_layer : str, optional
            Layer containing protein expression data
        protein_names : list, optional
            Specific proteins to analyze
        inplace : bool, default=True
            Whether to modify adata in place
        output_layer : str, default='protein_denoised'
            Name of output layer for denoised data
            
        Returns:
        --------
        adata_denoised : AnnData or None
            Denoised data (if inplace=False)
        """
        self.fit(adata, protein_layer, protein_names)
        return self.transform(adata, protein_layer, inplace, output_layer)
    
    def plot_protein_distribution(self, 
                                 protein_name: str, 
                                 adata: AnnData,
                                 protein_layer: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot protein distribution with fitted GMM and threshold.
        
        Parameters:
        -----------
        protein_name : str
            Name of protein to plot
        adata : AnnData
            Annotated data object
        protein_layer : str, optional
            Layer containing protein expression data
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.Figure
            Plot figure
        """
        if protein_name not in self.fitted_models_:
            raise ValueError(f"Protein {protein_name} not found in fitted models")
        
        # Get data
        protein_data = self._validate_adata(adata, protein_layer)
        protein_names = list(self.thresholds_.keys())
        protein_idx = protein_names.index(protein_name)
        protein_values = protein_data[:, protein_idx]
        
        # Get fitted model and threshold
        gmm = self.fitted_models_[protein_name]
        threshold = self.thresholds_[protein_name]
        stats = self.fit_stats_[protein_name]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Original distribution
        valid_values = protein_values[protein_values > 0]
        ax1.hist(valid_values, bins=50, density=True, alpha=0.7, color='lightblue')
        ax1.axvline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
        ax1.set_xlabel('Expression Level')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{protein_name} - Original Scale')
        ax1.legend()
        
        # Plot 2: Log-transformed with GMM components
        log_values = np.log1p(valid_values)
        ax2.hist(log_values, bins=50, density=True, alpha=0.7, color='lightblue')
        
        # Plot GMM components
        x_range = np.linspace(log_values.min(), log_values.max(), 1000)
        
        for i in range(self.n_components):
            mean = stats["means"][i]
            if self.covariance_type == 'full':
                std = np.sqrt(stats["covariances"][i][0, 0])
            elif self.covariance_type == 'diag':
                std = np.sqrt(stats["covariances"][i][0])
            else:
                std = np.sqrt(stats["covariances"][i])
            weight = stats["weights"][i]
            
            component_pdf = weight * stats.norm.pdf(x_range, mean, std)
            ax2.plot(x_range, component_pdf, 
                    label=f'Component {i+1} (w={weight:.2f})')
        
        ax2.axvline(np.log1p(threshold), color='red', linestyle='--',
                   label=f'Threshold (log): {np.log1p(threshold):.2f}')
        ax2.set_xlabel('Log(Expression + 1)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'{protein_name} - Log Scale with GMM')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_threshold_summary(self) -> pd.DataFrame:
        """
        Get summary of all calculated thresholds.
        
        Returns:
        --------
        summary_df : pd.DataFrame
            Summary table with thresholds and fit statistics
        """
        if not self.thresholds_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        summary_data = []
        for protein_name, threshold in self.thresholds_.items():
            if protein_name in self.fit_stats_:
                stats = self.fit_stats_[protein_name]
                summary_data.append({
                    'protein': protein_name,
                    'threshold': threshold,
                    'converged': stats.get('converged', False),
                    'n_valid_cells': stats.get('n_valid', 0),
                    'aic': stats.get('aic', np.nan),
                    'bic': stats.get('bic', np.nan),
                    'log_likelihood': stats.get('log_likelihood', np.nan)
                })
        
        return pd.DataFrame(summary_data)
    
    def save_thresholds(self, filepath: str):
        """
        Save calculated thresholds to CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save CSV file
        """
        summary_df = self.get_threshold_summary()
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Thresholds saved to {filepath}")


# Convenience functions for ScanPy integration
def pp_threshold_proteins(adata: AnnData,
                         protein_layer: Optional[str] = None,
                         protein_names: Optional[List[str]] = None,
                         n_components: int = 2,
                         inplace: bool = True,
                         output_layer: str = 'protein_denoised',
                         copy: bool = False) -> Optional[AnnData]:
    """
    Preprocess protein data using ThresholdPy (scanpy-style function).
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing CITE-seq data
    protein_layer : str, optional
        Layer containing protein expression data
    protein_names : list, optional
        Specific proteins to analyze
    n_components : int, default=2
        Number of GMM components
    inplace : bool, default=True
        Whether to modify adata in place
    output_layer : str, default='protein_denoised'
        Name of output layer for denoised data
    copy : bool, default=False
        Return a copy instead of writing to adata
        
    Returns:
    --------
    adata_denoised : AnnData or None
        Denoised data (if copy=True)
    """
    adata_work = adata.copy() if copy else adata
    
    # Initialize and fit ThresholdPy
    threshold_model = ThresholdPy(n_components=n_components)
    threshold_model.fit_transform(
        adata_work, 
        protein_layer=protein_layer,
        protein_names=protein_names,
        inplace=inplace,
        output_layer=output_layer
    )
    
    # Store model in uns for later access
    adata_work.uns['threshold_model'] = threshold_model
    
    return adata_work if copy else None


# Example usage and testing
if __name__ == "__main__":
    # Example with synthetic data
    import scanpy as sc
    
    # Create synthetic CITE-seq data
    n_cells = 1000
    n_proteins = 10
    
    # Create AnnData object
    adata = sc.AnnData(np.random.negative_binomial(5, 0.3, (n_cells, n_proteins)))
    adata.var_names = [f'Protein_{i}' for i in range(n_proteins)]
    adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
    
    # Add some noise and signal structure
    # Proteins 0-4: high noise, low signal
    # Proteins 5-9: low noise, high signal
    noise_proteins = adata.X[:, :5]
    signal_proteins = adata.X[:, 5:]
    
    # Add noise component
    noise_proteins += np.random.exponential(2, noise_proteins.shape)
    
    # Add signal component to expressing cells (50% of cells)
    expressing_cells = np.random.choice(n_cells, n_cells//2, replace=False)
    signal_proteins[expressing_cells] += np.random.exponential(10, (len(expressing_cells), 5))
    
    adata.X[:, :5] = noise_proteins
    adata.X[:, 5:] = signal_proteins
    
    # Apply ThresholdPy
    print("Applying ThresholdPy to synthetic data...")
    pp_threshold_proteins(adata, copy=False)
    
    # Get results
    threshold_model = adata.uns['threshold_model']
    summary = threshold_model.get_threshold_summary()
    print("\nThreshold Summary:")
    print(summary)
    
    # Plot example protein
    fig = threshold_model.plot_protein_distribution('Protein_0', adata)
    plt.show()
    
    print("\nDenoised data stored in adata.layers['protein_denoised']")
    print(f"Original data shape: {adata.X.shape}")
    print(f"Denoised data shape: {adata.layers['protein_denoised'].shape}")
