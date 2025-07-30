from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thresholdpy",
    version="0.1.5",
    author="Ross Jones",
    author_email="jonesr18@gmail.com",
    description="A Python adaptation of ThresholdR for CITE-seq denoising with ScanPy integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonesr18/thresholdpy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scanpy>=1.8.0",
        "anndata>=0.8.0",
        "igraph>=0.7.1",
        "leidenalg>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "mudata": [
            "mudata>=0.1.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
    },
    keywords="cite-seq, single-cell, denoising, gaussian-mixture-model, scanpy",
    project_urls={
        "Bug Reports": "https://github.com/jonesr18/thresholdpy/issues",
        "Source": "https://github.com/jonesr18/thresholdpy",
        "Documentation": "https://thresholdpy.readthedocs.io/",
    },
)
