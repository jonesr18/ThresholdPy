"""
Test suite for ThresholdPy package
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt

# Try to import MuData
try:
    from mudata import MuData
    HAS_MUDATA = True
except ImportError:
    HAS_MUDATA = False

from thresholdpy import ThresholdPy, pp_threshold_proteins


class TestThresholdPy:
    """Test cases for ThresholdPy class"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic CITE-seq data for testing"""
        np.random.seed(42)
        n_cells = 500
        n_proteins = 5
        
        # Create base data
        X = np.random.negative_binomial(3, 0.3, (n_cells, n_proteins)).astype(float)
        
        # Add noise and signal structure
        for i in range(n_proteins):
            # Add noise component (all cells)
            X[:, i] += np.random.exponential(1, n_cells)
            
            # Add signal component (subset of cells)
            expressing_cells = np.random.choice(
                n_cells, n_cells//3, replace=False
            )
            X[expressing_cells, i] += np.random.exponential(5, len(expressing_cells))
        
        # Create AnnData object
        adata = AnnData(X)
        adata.var_names = [f'Protein_{i}' for i in range(n_proteins)]
        adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Add to layer as well
        adata.layers['protein_raw'] = X.copy()
        
        return adata
        
    def test_initialization(self):
        """Test ThresholdPy initialization"""
        model = ThresholdPy()
        assert model.n_components == 2
        assert model.max_iter == 100
        assert model.random_state == 42
        assert model.covariance_type == 'full'
        assert len(model.thresholds_) == 0
        
        # Test custom parameters
        model_custom = ThresholdPy(
            n_components=3,
            max_iter=200,
            random_state=123,
            covariance_type='diag'
        )
        assert model_custom.n_components == 3
        assert model_custom.max_iter == 200
        assert model_custom.random_state == 123
        assert model_custom.covariance_type == 'diag'
    
    def test_validate_adata(self, synthetic_data):
        """Test AnnData validation"""
        model = ThresholdPy()
        
        # Test with .X
        _, protein_data = model._validate_adata(synthetic_data)
        assert protein_data.shape == synthetic_data.X.shape
        
        # Test with layer
        _, protein_data = model._validate_adata(synthetic_data, 'protein_raw')
        assert protein_data.shape == synthetic_data.layers['protein_raw'].shape

        # Test with specific protein
        _, protein_data = model._validate_adata(synthetic_data, 'protein_raw', ['Protein_0'])
        assert protein_data.flatten().shape == synthetic_data.layers['protein_raw'][:, 0].shape # flatten needed to force 1D in this case
        
        # Test error cases
        with pytest.raises(TypeError):
            model._validate_adata("not_anndata")
        
        with pytest.raises(ValueError):
            model._validate_adata(synthetic_data, 'nonexistent_layer')
    
    def test_fit(self, synthetic_data):
        """Test model fitting"""
        model = ThresholdPy()
        model.fit(synthetic_data)
        
        # Check that thresholds were calculated
        assert len(model.thresholds_) == synthetic_data.n_vars
        assert len(model.fitted_models_) <= synthetic_data.n_vars
        assert len(model.fit_stats_) <= synthetic_data.n_vars
        
        # Check that thresholds are reasonable
        for protein_name, threshold in model.thresholds_.items():
            if not np.isnan(threshold):
                assert threshold > 0
                assert threshold < synthetic_data.X.max()
    
    def test_fit_with_layer(self, synthetic_data):
        """Test fitting with specific layer"""
        model = ThresholdPy()
        model.fit(synthetic_data, protein_layer='protein_raw')
        
        assert len(model.thresholds_) == synthetic_data.n_vars
    
    def test_fit_with_protein_subset(self, synthetic_data):
        """Test fitting with protein subset"""
        model = ThresholdPy()
        protein_subset = ['Protein_0', 'Protein_2']
        model.fit(synthetic_data, protein_names=protein_subset)
        
        assert len(model.thresholds_) == len(protein_subset)
        for protein in protein_subset:
            assert protein in model.thresholds_
    
    def test_transform(self, synthetic_data):
        """Test data transformation"""
        model = ThresholdPy()
        model.fit(synthetic_data)
        
        # Test copy transformation
        adata_copy = model.transform(synthetic_data, inplace=False)
        assert isinstance(adata_copy, AnnData)
        assert 'protein_denoised' in adata_copy.layers
        assert 'protein_denoised' not in synthetic_data.layers

        with pytest.raises(ValueError, match="already exists"):
            model.transform(synthetic_data, output_layer='protein_raw', inplace=False)
        
        # Test inplace transformation
        model.transform(synthetic_data, inplace=True)
        assert 'protein_denoised' in synthetic_data.layers
        
        # Check that denoised data has more zeros
        original_zeros = np.sum(synthetic_data.X == 0)
        denoised_zeros = np.sum(synthetic_data.layers['protein_denoised'] == 0)
        assert denoised_zeros >= original_zeros
    
    def test_transform_without_fit(self, synthetic_data):
        """Test that transform raises error without fitting"""
        model = ThresholdPy()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.transform(synthetic_data)
    
    def test_fit_transform(self, synthetic_data):
        """Test combined fit and transform"""
        model = ThresholdPy()
        model.fit_transform(synthetic_data, inplace=True)
        
        assert len(model.thresholds_) == synthetic_data.n_vars
        assert 'protein_denoised' in synthetic_data.layers
    
    def test_get_threshold_summary(self, synthetic_data):
        """Test threshold summary generation"""
        model = ThresholdPy()
        model.fit(synthetic_data)
        
        summary = model.get_threshold_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(model.thresholds_)
        
        expected_columns = [
            'protein', 'threshold', 'converged', 'n_valid_cells',
            'aic', 'bic', 'log_likelihood'
        ]
        for col in expected_columns:
            assert col in summary.columns
    
    def test_get_threshold_summary_without_fit(self):
        """Test that summary raises error without fitting"""
        model = ThresholdPy()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_threshold_summary()
    
    def test_plot_protein_distribution(self, synthetic_data):
        """Test protein distribution plotting"""
        model = ThresholdPy()
        model.fit(synthetic_data)
        
        fig = model.plot_protein_distribution('Protein_0', synthetic_data, scales = ['log1p', 'sqrt', 'none'])
        assert isinstance(fig, plt.Figure)
        
        # Check that plot has expected structure
        assert len(fig.axes) == 3  # Three subplots
        
        # Test error for non-existent protein
        with pytest.raises(ValueError, match="not found in fitted models"):
            model.plot_protein_distribution('NonExistent', synthetic_data)
    
    def test_save_thresholds(self, synthetic_data, tmp_path):
        """Test saving thresholds to file"""
        model = ThresholdPy()
        model.fit(synthetic_data)
        
        filepath = tmp_path / "thresholds.csv"
        model.save_thresholds(str(filepath))
        
        # Check that file was created and has correct content
        assert filepath.exists()
        
        loaded_df = pd.read_csv(filepath)
        expected_df = model.get_threshold_summary()
        pd.testing.assert_frame_equal(loaded_df, expected_df)


class TestMuDataCompatibility:
    """Test MuData compatibility"""
    
    @pytest.fixture
    def synthetic_mudata(self):
        """Create a MuData object with RNA and protein modalities for testing"""
        if not HAS_MUDATA:
            pytest.skip("MuData not available")
            
        np.random.seed(42)
        n_cells = 500
        n_proteins = 5
        n_genes = 100
        
        # Create a mock RNA modality
        rna_data = np.random.negative_binomial(5, 0.1, (n_cells, n_genes))
        
        # Create AnnData for RNA
        adata_rna = AnnData(rna_data)
        adata_rna.var_names = [f'Gene_{i}' for i in range(n_genes)]
        adata_rna.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Create mock protein modality
        X = np.random.negative_binomial(3, 0.3, (n_cells, n_proteins)).astype(float)
        
        # Add noise and signal structure
        for i in range(n_proteins):
            # Add noise component (all cells)
            X[:, i] += np.random.exponential(1, n_cells)
            
            # Add signal component (subset of cells)
            expressing_cells = np.random.choice(
                n_cells, n_cells//3, replace=False
            )
            X[expressing_cells, i] += np.random.exponential(5, len(expressing_cells))
        
        # Create AnnData object
        adata_protein = AnnData(X)
        adata_protein.var_names = [f'Protein_{i}' for i in range(n_proteins)]
        adata_protein.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Add to layer as well
        adata_protein.layers['protein_raw'] = X.copy()

        # Create MuData with both modalities
        mudata = MuData({
            'rna': adata_rna,
            'prot': adata_protein},
            axis = 0
        )
        
        return mudata
    
    def test_mudata_auto_detection(self, synthetic_mudata):
        """Test automatic detection of protein modality in MuData"""
        if not HAS_MUDATA:
            pytest.skip("MuData not available")
            
        model = ThresholdPy()
        model.fit(synthetic_mudata)
        
        # Should have processed all proteins
        assert len(model.thresholds_) == synthetic_mudata['prot'].n_vars
        
        # Check that thresholds were calculated
        for protein in synthetic_mudata['prot'].var_names:
            assert protein in model.thresholds_
    
    def test_mudata_explicit_modality(self, synthetic_mudata):
        """Test explicit specification of protein modality"""
        if not HAS_MUDATA:
            pytest.skip("MuData not available")
            
        model = ThresholdPy()
        model.fit(synthetic_mudata, protein_modality='prot')
        
        # Should have processed all proteins
        assert len(model.thresholds_) == synthetic_mudata['prot'].n_vars
    
    def test_mudata_transform(self, synthetic_mudata):
        """Test transform with MuData input"""
        if not HAS_MUDATA:
            pytest.skip("MuData not available")
            
        model = ThresholdPy()
        model.fit(synthetic_mudata)
        
        # Test copy transformation
        mudata_copy = model.transform(synthetic_mudata, inplace=False)
        assert isinstance(mudata_copy, type(synthetic_mudata))
        assert 'protein_denoised' in mudata_copy['prot'].layers
        
        # Test inplace transformation - done second so that 'protein_denoised' layer is not added twice
        model.transform(synthetic_mudata, inplace=True)
        assert 'protein_denoised' in synthetic_mudata['prot'].layers
    
    def test_pp_function_with_mudata(self, synthetic_mudata):
        """Test the convenience function with MuData"""
        if not HAS_MUDATA:
            pytest.skip("MuData not available")
            
        # Test with automatic modality detection and adata copying
        result = pp_threshold_proteins(
            synthetic_mudata,
            n_components=2,
            inplace=False
        )
        assert isinstance(result, type(synthetic_mudata))
        assert 'protein_denoised' not in synthetic_mudata['prot'].layers
        assert 'protein_denoised' in result['prot'].layers
        
        # Test with explicit modality and inplace editing
        pp_threshold_proteins(
            synthetic_mudata,
            protein_modality='prot',
            n_components=2,
            inplace=True
        )
        assert 'protein_denoised' in synthetic_mudata['prot'].layers


class TestConvenienceFunctions:
    """Test convenience functions for ScanPy integration"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data (same as above)"""
        np.random.seed(42)
        n_cells = 200
        n_proteins = 3
        
        X = np.random.negative_binomial(3, 0.3, (n_cells, n_proteins)).astype(float)
        
        for i in range(n_proteins):
            X[:, i] += np.random.exponential(1, n_cells)
            expressing_cells = np.random.choice(
                n_cells, n_cells//3, replace=False
            )
            X[expressing_cells, i] += np.random.exponential(3, len(expressing_cells))
        
        adata = AnnData(X)
        adata.var_names = [f'Protein_{i}' for i in range(n_proteins)]
        adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        adata.layers['protein_raw'] = X.copy()
        
        return adata
    
    def test_pp_threshold_proteins_inplace(self, synthetic_data):
        """Test preprocessing function with inplace=True"""
        original_shape = synthetic_data.X.shape
        
        pp_threshold_proteins(synthetic_data, inplace=True)
        
        assert 'protein_denoised' in synthetic_data.layers
        assert synthetic_data.layers['protein_denoised'].shape == original_shape
        assert 'threshold_model' in synthetic_data.uns
    
    def test_pp_threshold_proteins_copy(self, synthetic_data):
        """Test preprocessing function with inplace=False"""
        result = pp_threshold_proteins(synthetic_data, inplace=False)
        
        assert isinstance(result, AnnData)
        assert 'protein_denoised' in result.layers
        assert 'threshold_model' in result.uns
        
        # Original should be unchanged
        assert 'protein_denoised' not in synthetic_data.layers
    
    def test_pp_threshold_proteins_with_layer(self, synthetic_data):
        """Test preprocessing function with specific layer"""
        pp_threshold_proteins(
            synthetic_data, 
            protein_layer='protein_raw',
            inplace=True
        )
        
        assert 'protein_denoised' in synthetic_data.layers
        assert 'threshold_model' in synthetic_data.uns
    
    def test_pp_threshold_proteins_with_subset(self, synthetic_data):
        """Test preprocessing function with protein subset"""
        protein_subset = ['Protein_0', 'Protein_1']
        
        pp_threshold_proteins(
            synthetic_data,
            protein_names=protein_subset,
            inplace=True
        )
        
        model = synthetic_data.uns['threshold_model']
        assert len(model.thresholds_) == len(protein_subset)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_data(self):
        """Test with empty data"""
        adata = AnnData(np.empty((0, 0)))
        model = ThresholdPy()
        
        # Should handle gracefully
        model.fit(adata)
        assert len(model.thresholds_) == 0
    
    def test_single_protein(self):
        """Test with single protein"""
        np.random.seed(42)
        X = np.random.exponential(2, (100, 1))
        # Add some high values to create bimodal distribution
        X[20:40, 0] += np.random.exponential(5, 20)
        
        adata = AnnData(X)
        adata.var_names = ['Protein_0']
        
        model = ThresholdPy()
        model.fit(adata)
        
        assert len(model.thresholds_) == 1
        assert 'Protein_0' in model.thresholds_
    
    def test_all_zero_protein(self):
        """Test with protein that has all zero values"""
        X = np.zeros((100, 2))
        X[:, 1] = np.random.exponential(2, 100)  # Only second protein has values
        
        adata = AnnData(X)
        adata.var_names = ['Zero_Protein', 'Normal_Protein']
        
        model = ThresholdPy()
        model.fit(adata, scale='log1p')
        
        # Should handle zero protein gracefully
        assert len(model.thresholds_) == 2
        assert np.isnan(model.thresholds_['Zero_Protein'])
        assert not np.isnan(model.thresholds_['Normal_Protein'])
    
    def test_insufficient_data(self):
        """Test with insufficient non-zero values"""
        X = np.zeros((100, 1))
        X[:5, 0] = [1, 2, 3, 4, 5]  # Only 5 non-zero values
        
        adata = AnnData(X)
        adata.var_names = ['Sparse_Protein']
        
        model = ThresholdPy()
        model.fit(adata, scale='log1p')
        
        # Should handle insufficient data
        assert len(model.thresholds_) == 1
        # Threshold should be NaN due to insufficient data
        assert np.isnan(model.thresholds_['Sparse_Protein'])


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
