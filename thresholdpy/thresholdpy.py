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
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import scanpy as sc
from anndata import AnnData
import logging

# Try to import MuData for multi-modal data support
try:
    from mudata import MuData
    HAS_MUDATA = True
except ImportError:
    HAS_MUDATA = False

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
        
    def _get_protein_mask(self, adata: AnnData) -> np.ndarray:
        """
        Get a boolean mask for protein features in AnnData.var
        
        Parameters:
        -----------
        adata : AnnData
            Annotated data object
            
        Returns:
        --------
        protein_mask : np.ndarray
            Boolean mask where True indicates protein features
        """
        if not hasattr(adata, 'var') or not hasattr(adata.var, 'columns'):
            # If no var annotations, assume all features are proteins
            return np.ones(adata.n_vars, dtype=bool)
            
        # Check common column names for protein annotations
        for col in ["feature_types", "modality"]:
            if col in adata.var.columns:
                logger.info(f"Using '{col}' column to identify protein features")
                return adata.var[col].astype(str).str.lower().isin(
                    ["antibody capture", "protein", "prot", "adt"]
                )
                
        # If no protein-specific columns found, assume all features are proteins
        return np.ones(adata.n_vars, dtype=bool)
    
    def _validate_adata(self,
                        adata: Any,
                        protein_layer: Optional[str] = None,
                        protein_names: Optional[List[str]] = None,
                        protein_modality: Optional[str] = None) -> Tuple[AnnData, np.ndarray]:
        """
        Validate and extract protein expression data from AnnData or MuData object.
        
        Parameters:
        -----------
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
            can be used to specify the protein modality.
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X.
        protein_names : list, optional
            List of specific protein names to extract. If None, uses all protein features.
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
            
        Returns:
        --------
        adata : AnnData
            The processed AnnData object (potentially subset to protein modality)
        protein_data : np.ndarray
            Protein expression matrix (cells x proteins)
        """

        # Handle AnnData vs MuData objects - extracts protein modality AnnData from MuData
        adata = self._handle_mudata(adata, protein_modality)
        
        # At this point, adata should be an AnnData object
        if not isinstance(adata, AnnData):
            raise TypeError(f"Expected AnnData or MuData, got {type(adata).__name__}")
        
        # Get protein mask
        self.protein_mask_ = self._get_protein_mask(adata)
        
        # Get protein data from the specified layer or .X as a default
        if protein_layer is None:
            protein_data = adata.X
        else:
            if protein_layer in adata.layers:
                protein_data = adata.layers[protein_layer]
            else:
                raise ValueError(f"Layer '{protein_layer}' not found in AnnData object")
            
        if protein_data is None:
            raise ValueError("No protein expression data found")
        
        # Convert to dense array if sparse
        if hasattr(protein_data, 'toarray'):
            protein_data = protein_data.toarray()
        
        # Subset to specific protein names if provided
        if protein_names is not None:
            protein_mask = np.isin(adata.var_names, protein_names)
            protein_data = protein_data[:, protein_mask]
        # Otherwise subset to protein features if not all features are proteins
        elif not np.all(self.protein_mask_):
            protein_data = protein_data[:, self.protein_mask_]
        
        return adata, protein_data

    def _handle_mudata(self, 
                       adata: Any, 
                       protein_modality: Optional[str] = None) -> AnnData:
        """
        Handle MuData objects by extracting the protein modality.
        
        Parameters:
        -----------
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data, can be either AnnData or MuData
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
            
        Returns:
        --------
        adata : AnnData
            AnnData object containing protein expression data
        """
        if HAS_MUDATA and isinstance(adata, MuData):
            if protein_modality is None:
                # Try to find protein modality automatically
                protein_modalities = [mod for mod in adata.mod if 'prot' in mod.lower() or 'protein' in mod.lower() or 'adt' in mod.lower()]
                if not protein_modalities:
                    raise ValueError("Could not automatically determine protein modality. "
                                   "Please specify protein_modality as the name of the protein modality.")
                if len(protein_modalities) > 1:
                    logger.warning(f"Multiple potential protein modalities found: {protein_modalities}")
                protein_modality = protein_modalities[0]
                logger.info(f"Using protein modality: {protein_modality}")
                adata = adata[protein_modality]
            elif protein_modality in adata.mod:
                adata = adata[protein_modality]
            else:
                raise ValueError(f"Modality '{protein_modality}' not found in MuData object")
        # Ensure AnnData object otherwise
        elif not isinstance(adata, AnnData):
            raise TypeError(f"Input must be an AnnData or MuData object, got {type(adata).__name__}")
        
        return adata
    
    def _fit_gmm_single_protein(self, 
                               protein_values: np.ndarray, 
                               protein_name: str,
                               transform: Optional[str] = 'none') -> Tuple[GaussianMixture, Dict]:
        """
        Fit GMM to a single protein's expression values.
        
        Parameters:
        -----------
        protein_values : np.ndarray
            Expression values for one protein
        protein_name : str
            Name of the protein
        transform : str, default='log1p'
            Transformation to apply to data before fitting GMM
            
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
        
        # Transform data
        if transform == None or transform.lower() == 'none':
            transf_values = valid_values.reshape(-1, 1)
        elif transform.lower() == 'log1p':
            transf_values = np.log1p(valid_values).reshape(-1, 1)
        elif transform.lower() == 'sqrt':
            transf_values = np.sqrt(valid_values).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            random_state=self.random_state,
            covariance_type=self.covariance_type
        )
        
        try:
            gmm.fit(transf_values)
            
            # Calculate statistics
            aic = gmm.aic(transf_values)
            bic = gmm.bic(transf_values)
            log_likelihood = gmm.score(transf_values)
            
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
    
    def _calculate_threshold(self, 
                             gmm: GaussianMixture, 
                             stats: Dict, 
                             transform: Optional[str] = None) -> float:
        """
        Calculate the threshold separating noise and signal components.
        
        Parameters:
        -----------
        gmm : GaussianMixture
            Fitted GMM model
        stats : dict
            Fitting statistics
        transform : str, optional
            Data transformation applied before fitting GMM. Options are 'none', 'log1p', 'sqrt'.
            
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
        threshold = noise_mean + 2 * noise_std
        
        # Convert back to original scale
        match transform:
            case 'log1p':
                threshold = np.expm1(threshold)
            case 'sqrt':
                threshold = threshold**2
            case _:
                threshold = threshold
        
        return threshold

    def fit(self, 
            adata: Union[AnnData, MuData], 
            protein_layer: Optional[str] = None,
            protein_names: Optional[List[str]] = None,
            protein_modality: Optional[str] = None,
            transform: Optional[str] = 'none') -> 'ThresholdPy':
        """
        Fit GMM models for each protein.
        
        Parameters:
        -----------
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
            can be used to specify the protein modality.
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X.
        protein_names : list, optional
            Specific proteins to analyze. If None, uses all proteins identified by feature annotations
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
            
        Returns:
        --------
        self : ThresholdPy
            Fitted ThresholdPy object
        """
        logger.info("Starting ThresholdPy fitting...")
        
        # Validate and extract data
        adata, protein_data = self._validate_adata(adata, protein_layer, protein_names, protein_modality)
        
        # Get protein names from var if not provided
        if protein_names is None:
            if hasattr(adata, 'var') and hasattr(adata.var, 'index'):
                # Use feature names from the protein features only
                if hasattr(self, 'protein_mask_'):
                    protein_names = adata.var.index[self.protein_mask_].tolist()
                else:
                    protein_names = adata.var.index.tolist()
            else:
                # Fallback to generic names
                protein_names = [f"Protein_{i+1}" for i in range(protein_data.shape[1])]
        
        # Ensure we have the correct number of protein names
        # - shouldn't be necessary anymore
        # if len(protein_names) != protein_data.shape[1]:
        #     logger.warning(
        #         f"Number of protein names ({len(protein_names)}) doesn't match "
        #         f"number of protein features ({protein_data.shape[1]}). Using default names."
        #     )
        #     protein_names = [f"Protein_{i+1}" for i in range(protein_data.shape[1])]
            
        logger.info(f"Fitting models for {len(protein_names)} of {protein_data.shape[1]} proteins")

        # Fit GMM for each protein
        for i, protein_name in enumerate(protein_names):
            logger.info(f"Fitting GMM for {protein_name} ({i+1}/{len(protein_names)})")

            protein_values = protein_data[:, i]
            gmm, stats = self._fit_gmm_single_protein(protein_values, protein_name, transform = transform)
            
            if gmm is not None:
                threshold = self._calculate_threshold(gmm, stats, transform = transform)
                self.thresholds_[protein_name] = threshold
                self.fitted_models_[protein_name] = gmm
                self.fit_stats_[protein_name] = stats
            else:
                logger.warning(f"Failed to fit GMM for {protein_name}")
                self.thresholds_[protein_name] = np.nan
        
        logger.info(f"Completed fitting for {len(self.thresholds_)} proteins")
        return self
    
    def transform(self, 
                  adata: Union[AnnData, MuData], 
                  protein_layer: Optional[str] = None,
                  protein_modality: Optional[str] = None,
                  inplace: bool = True,
                  output_layer: str = 'protein_denoised') -> Optional[Union[AnnData, MuData]]:
        """
        Apply thresholds to denoise protein expression data.
        
        Parameters:
        -----------
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
            can be used to specify the protein modality.
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X.
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
        inplace : bool, default=True
            Whether to modify adata in place
        output_layer : str, default='protein_denoised'
            Name of output layer for denoised data
        transform : str, optional
            Data transformation to apply before fitting GMM. Options are 'none', 'log1p', 'sqrt'.
            
        Returns:
        --------
        adata_denoised : AnnData or MuData or None
            Denoised data (if inplace=False)
        """
        if not self.thresholds_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Validate and extract data
        _, protein_data = self._validate_adata(adata, protein_layer, protein_modality = protein_modality)
        
        # Apply thresholds
        # denoised_data = protein_data.copy()
        denoised_data = np.full((protein_data.shape[0], len(self.protein_mask_)), np.nan) # Account for non-protein features
        protein_idxs = np.where(self.protein_mask_)[0]

        protein_names = list(self.thresholds_.keys())
        for i, protein_name in enumerate(protein_names):
            if i >= protein_data.shape[1]:
                break
                
            threshold = self.thresholds_[protein_name]
            if not np.isnan(threshold):
                # Set values below threshold to 0
                denoised_data[:, protein_idxs[i]] = np.where(
                    protein_data[:, i] < threshold, 0, protein_data[:, i]
                )

        # Similar to _handle_mudata but for writing to mudata vs anndata
        if HAS_MUDATA and isinstance(adata, MuData):
            if protein_modality is None:
                # Try to find protein modality automatically
                protein_modalities = [mod for mod in adata.mod if 'prot' in mod.lower() or 'protein' in mod.lower() or 'adt' in mod.lower()]
                if not protein_modalities:
                    raise ValueError("Could not automatically determine protein modality. "
                                   "Please specify protein_modality as the name of the protein modality.")
                if len(protein_modalities) > 1:
                    logger.warning(f"Multiple potential protein modalities found: {protein_modalities}")
                protein_modality = protein_modalities[0]
                logger.info(f"Writing to protein modality: {protein_modality}")
            elif protein_modality not in adata.mod:
                raise ValueError(f"Modality '{protein_modality}' not found in MuData object")
            
            if output_layer in adata[protein_modality].layers:
                raise ValueError(f"Output layer {output_layer} already exists. Use a different output_layer name to avoid overwriting.")
            
            if inplace:
                adata[protein_modality].layers[output_layer] = denoised_data
                return adata
            else:
                adata_copy = adata.copy()
                adata_copy[protein_modality].layers[output_layer] = denoised_data
                return adata_copy
        
        # Handle AnnData objects
        elif isinstance(adata, AnnData):
            if output_layer in adata.layers:
                raise ValueError(f"Output layer {output_layer} already exists. Use a different output_layer name to avoid overwriting.")
            
            if inplace:
                adata.layers[output_layer] = denoised_data
            else:
                adata_copy = adata.copy()
                adata_copy.layers[output_layer] = denoised_data
                return adata_copy
    
    def fit_transform(self, 
                     adata: Union[AnnData, MuData], 
                     protein_layer: Optional[str] = None,
                     protein_names: Optional[List[str]] = None,
                     protein_modality: Optional[str] = None,
                     inplace: bool = True,
                     output_layer: str = 'protein_denoised',
                     transform: Optional[str] = 'none') -> Optional[Union[AnnData, MuData]]:
        """
        Fit GMM models and apply thresholds in one step.
        
        Parameters:
        -----------
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
            can be used to specify the protein modality.
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X.
        protein_names : list, optional
            Specific proteins to analyze
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
        inplace : bool, default=True
            Whether to modify adata in place
        output_layer : str, default='protein_denoised'
            Name of output layer for denoised data
        transform : str, optional
            Data transformation to apply before fitting GMM. Options are 'none', 'log1p', 'sqrt'.
            
        Returns:
        --------
        adata_denoised : AnnData or MuData or None
            Denoised data (if inplace=False)
        """
        self.fit(adata, protein_layer, protein_names, protein_modality, transform = transform)
        return self.transform(adata, protein_layer, protein_modality, inplace, output_layer)
    
    def plot_protein_distribution(self, 
                                 protein_name: str, 
                                 adata: Union[AnnData, MuData],
                                 protein_layer: Optional[str] = None,
                                 protein_modality: Optional[str] = None,
                                 transforms: Optional[List[str]] = ['none'],
                                 figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot protein distribution with fitted GMM and threshold.
        
        Parameters:
        -----------
        protein_name : str
            Name of protein to plot
        adata : Union[AnnData, MuData]
            Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
            can be used to specify the protein modality.
        protein_layer : str, optional
            Layer containing protein expression data. If None, uses .X.
        protein_modality : str, optional
            Name of the protein modality in MuData. If None, uses the first protein modality found.
        figsize : tuple, default=(5*len(transforms), 6)
            Figure size
        transforms : list, optional
            Data transformation(s) for plots. Options are 'none', 'log1p', 'sqrt'.
            
        Returns:
        --------
        fig : matplotlib.Figure
            Plot figure
        """
        if protein_name not in self.fitted_models_:
            raise ValueError(f"Protein {protein_name} not found in fitted models")
        
        # Get data
        _, protein_data = self._validate_adata(adata, protein_layer, protein_name, protein_modality)
        
        # Get fitted model and threshold
        gmm = self.fitted_models_[protein_name]
        threshold = self.thresholds_[protein_name]
        fit_stats = self.fit_stats_[protein_name]
        
        # Create plot
        if figsize is None:
            figsize = (5*len(transforms), 6)
        fig, axs = plt.subplots(1, len(transforms), figsize=figsize)
        if len(transforms) == 1:
            axs = [axs] # ensure list for iteration

        # Plot 1: Original distribution
        # valid_values = protein_data[protein_data > 0]
        for transf, ax in zip(transforms, axs):
            match transf:
                case 'log1p':
                    transf_data = np.log1p(protein_data)
                case 'sqrt':
                    transf_data = np.sqrt(protein_data)
                case 'none':
                    transf_data = protein_data.copy()
            ax.hist(transf_data, bins=50, density=True, alpha=0.7, color='lightblue')
            ax.axvline(threshold, color='red', linestyle='--', 
                    label=f'Threshold: {threshold:.2f}')
            ax.set_xlabel('Expression Level')
            ax.set_ylabel('Density')
            ax.set_title(f'{protein_name} - Transf: {transf}')
            ax.legend()
        
            # Plot GMM components
            x_range = np.linspace(transf_data.min(), transf_data.max(), 1000)
            
            for i in range(self.n_components):
                mean = fit_stats["means"][i]
                if self.covariance_type == 'full':
                    std = np.sqrt(fit_stats["covariances"][i][0, 0])
                elif self.covariance_type == 'diag':
                    std = np.sqrt(fit_stats["covariances"][i][0])
                else:
                    std = np.sqrt(fit_stats["covariances"][i])
                weight = fit_stats["weights"][i]

                component_pdf = weight * stats.norm.pdf(x_range, mean, std)
                ax.plot(x_range, component_pdf, 
                        label=f'Component {i+1} (w={weight:.2f})')
        
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
def pp_threshold_proteins(adata: Union[AnnData, MuData],
                         protein_layer: Optional[str] = None,
                         protein_names: Optional[List[str]] = None,
                         protein_modality: Optional[str] = None,
                         n_components: int = 2,
                         inplace: bool = True,
                         output_layer: str = 'protein_denoised',
                         transform: Optional[str] = 'none') -> Optional[Union[AnnData, MuData]]:
    """
    Preprocess protein data using ThresholdPy (scanpy-style function).
    
    Parameters:
    -----------
    adata : Union[AnnData, MuData]
        Annotated data object containing CITE-seq data. If MuData, optional input protein_modality
        can be used to specify the protein modality.
    protein_layer : str, optional
        Layer containing protein expression data. If None, uses .X.
    protein_names : list, optional
        Specific proteins to analyze
    protein_modality : str, optional
        Name of the protein modality in MuData. If None, uses the first protein modality found.
    n_components : int, default=2
        Number of GMM components
    inplace : bool, default=True
        Whether to modify adata in place
    output_layer : str, default='protein_denoised'
        Name of output layer for denoised data
    transform : str, optional
        Data transformation to apply before fitting GMM. Options are 'none', 'log1p', 'sqrt'.
    
    Returns:
    --------
    adata_denoised : AnnData or MuData or None
        Denoised data (a copy of adata input if inplace=False)
    """

    # Initialize and fit ThresholdPy
    threshold_model = ThresholdPy(n_components=n_components)
    adata_denoised = threshold_model.fit_transform(
        adata, 
        protein_layer=protein_layer,
        protein_names=protein_names,
        protein_modality=protein_modality,
        inplace=inplace,
        output_layer=output_layer,
        transform = transform
    )
    
    # Store model in uns for later access
    if inplace:
        adata.uns['threshold_model'] = threshold_model
        return adata
    else:
        adata_denoised.uns['threshold_model'] = threshold_model
        return adata_denoised


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
    pp_threshold_proteins(adata, inplace=True)
    
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
