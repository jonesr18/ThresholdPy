# ThresholdPy

A Python adaptation of [ThresholdR](https://github.com/MDMotlagh/ThresholdR) for CITE-seq denoising with ScanPy integration.

## Overview

ThresholdPy uses Gaussian Mixture Models (GMM) to identify noise populations in surface markers across cells in CITE-seq experiments. Once the noise distribution is identified, it calculates the upper threshold of the noise component to separate expressing and non-expressing cells for each surface marker.

## Key Features

- **Gaussian Mixture Model-based denoising**: Automatically identifies noise and signal components
- **ScanPy integration**: Works seamlessly with AnnData objects
- **Flexible protein selection**: Analyze all proteins or specify subsets
- **Comprehensive visualization**: Plot protein distributions with fitted models
- **Threshold export**: Save calculated thresholds for reproducibility
- **Memory efficient**: Handles large single-cell datasets

## Installation

```bash
pip install thresholdpy
```

Or install from source:

```bash
git clone https://github.com/yourusername/thresholdpy.git
cd thresholdpy
pip install -e .
```

## Quick Start

```python
import scanpy as sc
import thresholdpy as tp

# Load your CITE-seq data
adata = sc.read_h5ad('your_citeseq_data.h5ad')

# Method 1: Use the scanpy-style preprocessing function
tp.pp_threshold_proteins(adata, protein_layer='protein_raw')

# Method 2: Use the ThresholdPy class directly
threshold_model = tp.ThresholdPy(n_components=2)
threshold_model.fit_transform(adata, protein_layer='protein_raw')

# Access denoised data
denoised_proteins = adata.layers['protein_denoised']

# Get threshold summary
summary = threshold_model.get_threshold_summary()
print(summary)
```

## Detailed Usage

### Basic Denoising

```python
import scanpy as sc
import thresholdpy as tp

# Load data
adata = sc.read_h5ad('citeseq_data.h5ad')

# Apply ThresholdPy denoising
tp.pp_threshold_proteins(
    adata,
    protein_layer='protein_raw',  # Layer containing raw protein counts
    output_layer='protein_denoised',  # Output layer name
    n_components=2,  # Number of GMM components
    inplace=True  # Modify adata in place
)

# The denoised data is now available in adata.layers['protein_denoised']
```

### Advanced Usage with Custom Parameters

```python
# Initialize with custom parameters
threshold_model = tp.ThresholdPy(
    n_components=3,  # Use 3 components if you expect multiple populations
    max_iter=200,    # More iterations for convergence
    covariance_type='tied',  # Use tied covariance
    random_state=123
)

# Fit to specific proteins only
protein_subset = ['CD3', 'CD4', 'CD8', 'CD19', 'CD56']
threshold_model.fit(
    adata, 
    protein_layer='protein_raw',
    protein_names=protein_subset
)

# Apply thresholds
threshold_model.transform(adata, output_layer='protein_denoised_subset')
```

### Visualization

```python
# Plot distribution for a specific protein
fig = threshold_model.plot_protein_distribution(
    'CD3', 
    adata, 
    protein_layer='protein_raw'
)

# Save the plot
fig.savefig('CD3_threshold_plot.pdf', dpi=300, bbox_inches='tight')
```

### Export Thresholds

```python
# Get threshold summary table
summary_df = threshold_model.get_threshold_summary()
print(summary_df)

# Save thresholds to CSV
threshold_model.save_thresholds('protein_thresholds.csv')
```

## Method Details

### Gaussian Mixture Model Approach

1. **Data Preprocessing**: Log-transform protein expression values to make them more Gaussian-like
2. **GMM Fitting**: Fit a 2-component (or more) Gaussian Mixture Model to each protein
3. **Component Identification**: Identify the noise component (typically with lower mean)
4. **Threshold Calculation**: Calculate threshold as mean + 2×std of the noise component
5. **Denoising**: Set values below threshold to zero

### Model Selection

The package automatically selects the best model based on:
- **Convergence**: Models that fail to converge are excluded
- **AIC/BIC**: Information criteria for model comparison
- **Biological plausibility**: Components with reasonable means and variances

## Parameters

### ThresholdPy Class Parameters

- `n_components` (int, default=2): Number of mixture components
- `max_iter` (int, default=100): Maximum EM iterations
- `random_state` (int, default=42): Random seed for reproducibility
- `covariance_type` (str, default='full'): GMM covariance type

### Function Parameters

- `protein_layer` (str, optional): AnnData layer containing protein data
- `protein_names` (list, optional): Specific proteins to analyze
- `inplace` (bool, default=True): Modify AnnData object in place
- `output_layer` (str, default='protein_denoised'): Output layer name

## Output

### Denoised Data
- Stored in `adata.layers[output_layer]`
- Values below calculated thresholds set to zero
- Maintains original data structure and cell annotations

### Threshold Information
- `get_threshold_summary()`: DataFrame with thresholds and fit statistics
- Includes convergence status, AIC/BIC values, and component parameters
- Can be exported to CSV for reproducibility

### Visualization
- Distribution plots showing original data, GMM components, and thresholds
- Both original and log-transformed scales
- Customizable figure size and styling

## Comparison with Original ThresholdR

| Feature | ThresholdR (R) | ThresholdPy (Python) |
|---------|----------------|----------------------|
| GMM Implementation | mixtools/mclust | scikit-learn |
| Data Structure | Seurat objects | AnnData objects |
| Visualization | ggplot2 | matplotlib/seaborn |
| Integration | Seurat ecosystem | ScanPy ecosystem |
| Performance | R-optimized | NumPy/SciPy optimized |
| Memory Usage | R memory model | Python memory model |

## Examples

### Example 1: Basic CITE-seq Processing

```python
import scanpy as sc
import thresholdpy as tp
import matplotlib.pyplot as plt

# Load CITE-seq data
adata = sc.read_10x_mtx('filtered_feature_bc_matrix/')
adata.var_names_unique()

# Separate RNA and protein data
rna_genes = adata.var_names.str.startswith('ENSG')
protein_genes = ~rna_genes

adata_rna = adata[:, rna_genes].copy()
adata_protein = adata[:, protein_genes].copy()

# Process RNA data (standard scanpy workflow)
sc.pp.filter_cells(adata_rna, min_genes=200)
sc.pp.filter_genes(adata_rna, min_cells=3)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)

# Process protein data with ThresholdPy
tp.pp_threshold_proteins(adata_protein, inplace=True)

# Combine back for joint analysis
adata_combined = adata_rna.concatenate(adata_protein, batch_key='modality')
```

### Example 2: Quality Control and Comparison

```python
# Apply ThresholdPy
threshold_model = tp.ThresholdPy()
threshold_model.fit_transform(adata, protein_layer='protein_raw')

# Compare before and after denoising
import numpy as np

original_data = adata.layers['protein_raw']
denoised_data = adata.layers['protein_denoised']

# Calculate sparsity
original_sparsity = np.mean(original_data == 0)
denoised_sparsity = np.mean(denoised_data == 0)

print(f"Original sparsity: {original_sparsity:.2%}")
print(f"Denoised sparsity: {denoised_sparsity:.2%}")

# Plot comparison for specific protein
protein_idx = 0
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(original_data[:, protein_idx], bins=50, alpha=0.7)
axes[0].set_title('Original')
axes[0].set_xlabel('Expression')

axes[1].hist(denoised_data[:, protein_idx], bins=50, alpha=0.7)
axes[1].set_title('Denoised')
axes[1].set_xlabel('Expression')
```

### Example 3: Batch Processing

```python
# Process multiple samples
samples = ['sample1.h5ad', 'sample2.h5ad', 'sample3.h5ad']
threshold_models = {}

for sample in samples:
    adata = sc.read_h5ad(sample)
    
    # Fit ThresholdPy model
    model = tp.ThresholdPy()
    model.fit_transform(adata, protein_layer='protein_raw')
    
    # Store model for later use
    threshold_models[sample] = model
    
    # Save processed data
    adata.write(f"processed_{sample}")

# Compare thresholds across samples
import pandas as pd

all_thresholds = []
for sample, model in threshold_models.items():
    summary = model.get_threshold_summary()
    summary['sample'] = sample
    all_thresholds.append(summary)

combined_thresholds = pd.concat(all_thresholds)
threshold_comparison = combined_thresholds.pivot(
    index='protein', 
    columns='sample', 
    values='threshold'
)
print(threshold_comparison)
```

## Troubleshooting

### Common Issues

1. **Convergence Failures**
   - Try increasing `max_iter`
   - Use simpler covariance type ('spherical' or 'diag')
   - Check for proteins with too few expressing cells

2. **Memory Issues**
   - Process proteins in batches using `protein_names` parameter
   - Use sparse data structures when possible
   - Consider downsampling very large datasets

3. **Poor Threshold Quality**
   - Visualize protein distributions to check GMM fit
   - Try different numbers of components
   - Consider log-transforming data beforehand

### Performance Tips

- Use `n_components=2` for most applications
- Set `random_state` for reproducible results
- Process subsets of proteins if memory is limited
- Use `inplace=True` to save memory

## Citation

If you use ThresholdPy in your research, please cite:

```bibtex
@software{thresholdpy2024,
  title={ThresholdPy: A Python adaptation of ThresholdR for CITE-seq denoising},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/thresholdpy}
}
```

And the original ThresholdR paper:
```bibtex
@article{motlagh2024thresholdr,
  title={ThresholdR: A Denoising Pipeline for ADT data from CITE-seq experiments},
  author={Motlagh, MD and others},
  year={2024},
  url={https://github.com/MDMotlagh/ThresholdR}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original ThresholdR package by MDMotlagh
- ScanPy development team for the excellent single-cell analysis framework
- scikit-learn for robust machine learning implementations
- Claude Sonnet 4 for initial conversion from R to Python
