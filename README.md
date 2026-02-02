# Machine Learning for Predictive Reynolds Stress Transport Modeling

Implementation of the research paper "Evaluation of Machine Learning Algorithms for Predictive Reynolds Stress Transport Modeling" by J.P. Panda and H.V. Warrior, Department of Ocean Engineering and Naval Architecture, Indian Institute of Technology Kharagpur.

## Abstract

This repository provides a complete implementation of data-driven turbulence modeling using machine learning algorithms applied to Reynolds Stress Transport Equations. The work addresses fundamental limitations of eddy viscosity models by directly modeling the pressure-strain correlation term using high-fidelity Direct Numerical Simulation (DNS) data. Three distinct machine learning approaches are evaluated: Artificial Neural Networks, Random Forests, and Gradient Boosted Decision Trees, with hyperparameter optimization performed via Bayesian methods.

## Research Context

### The Turbulence Modeling Challenge

Computational Fluid Dynamics relies heavily on turbulence models, but traditional eddy viscosity models fail in complex flows involving streamline curvature, flow separation, system rotation, and secondary flows. Reynolds Stress Transport Models (RSTM) overcome these limitations by solving transport equations for each Reynolds stress component, explicitly accounting for turbulence anisotropy and directional effects.

### The Closure Problem

The critical challenge in RSTM is modeling the pressure-strain correlation term, which governs energy redistribution between Reynolds stress components. Classical physics-based models have significant limitations in complex engineering flows. This work demonstrates that machine learning can learn these complex relationships directly from DNS data, potentially generalizing to new flow configurations.

## Project Structure

The implementation is organized into seven sequential Jupyter notebooks:

### Phase 0: Introduction and Overview
Comprehensive introduction to turbulence modeling hierarchy, Reynolds stress transport framework, the pressure-strain correlation closure problem, and project objectives. Establishes the theoretical foundation and presents the leave-one-out cross-validation strategy.

### Phase 1: Physics-Informed Feature Engineering and Dataset Development
Extraction and processing of DNS data from the Lee & Moser turbulent channel flow database. Computation of physical features including Reynolds stress anisotropy, turbulent kinetic energy, dissipation rate, and mean velocity gradients. Construction of training datasets for four Reynolds numbers (Re_tau = 550, 1000, 2000, 5200).

### Phase 2: Exploratory Data Analysis
Statistical analysis of turbulence parameters across different Reynolds numbers. Visualization of turbulence structure, including log-layer behavior, Reynolds stress distributions, and anisotropy evolution. Assessment of data quality and identification of physical constraints.

### Phase 3: Machine Learning Model Development and Hyperparameter Optimization
Implementation of three regression algorithms:
- Multi-Layer Perceptron (PyTorch)
- Random Forest (Scikit-learn)
- Gradient Boosted Decision Trees (XGBoost)

Bayesian hyperparameter optimization using Optuna. Training on leave-one-out configurations to assess model robustness across different Reynolds numbers.

### Phase 4: Model Testing and Evaluation
Comprehensive evaluation of trained models on held-out channel flow data. Analysis of prediction accuracy, error distributions, and model performance metrics. Comparison of algorithm strengths and limitations. Feature importance analysis to validate physics-based learning.

### Phase 5: Out-of-Distribution Testing on Couette Flow
Critical generalization test using turbulent Couette flow, which is driven by wall motion rather than pressure gradient. This tests whether models learned fundamental turbulence physics or merely memorized channel flow patterns. Validation of model transferability to flows with different forcing mechanisms.

### Phase 6: Comprehensive Summary and Conclusions
Synthesis of results across all experiments. Discussion of algorithm performance, generalization capabilities, and practical implications for CFD applications. Identification of best practices and recommendations for data-driven turbulence modeling.

## Dataset

### Training Data: Turbulent Channel Flow
High-fidelity DNS data from the Oden Institute Turbulence File Server (Lee & Moser, 2015). Four friction Reynolds numbers: Re_tau = 550, 1000, 2000, 5200. Pressure-driven flow between parallel plates with fully resolved turbulent scales.

### Testing Data: Turbulent Couette Flow
DNS data at Re_tau approximately 500 (Lee & Moser, 2018). Shear-driven flow with moving walls and no pressure gradient. Provides independent test of model generalization to different flow physics.

### Leave-One-Out Cross-Validation Strategy

| Case | Training Set        | Testing Set |
|------|---------------------|-------------|
| 1    | 550, 1000, 2000     | 5200        |
| 2    | 550, 1000, 5200     | 2000        |
| 3    | 550, 2000, 5200     | 1000        |
| 4    | 1000, 2000, 5200    | 550         |

## Methodology

### Functional Mapping
The pressure-strain correlation is approximated as:
```
phi_uv = f(b_uv, epsilon, dU/dy, k)
```
where b_uv is Reynolds stress anisotropy, epsilon is dissipation rate, dU/dy is mean velocity gradient, and k is turbulent kinetic energy.

### Input Features
Physics-informed features derived from classical turbulence modeling:
- Reynolds stress anisotropy tensor components
- Turbulent kinetic energy
- Dissipation rate
- Mean velocity gradient (strain rate)

All features normalized to [0, 1] range to improve training convergence.

### Model Architectures

**Artificial Neural Network:**
- Multi-layer perceptron with 5 hidden layers
- 10 neurons per layer (manual tuning)
- ReLU activation functions
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting

**Random Forest:**
- Ensemble of decision trees trained on bootstrap samples
- Maximum depth and number of estimators optimized
- Feature importance analysis capability
- Inherent resistance to overfitting through averaging

**Gradient Boosted Decision Trees:**
- Sequential ensemble correcting residual errors
- XGBoost implementation for computational efficiency
- Extensive hyperparameter space: depth, estimators, learning rate, sample splits
- Bayesian optimization for hyperparameter tuning

### Hyperparameter Optimization
Bayesian optimization via Gaussian Process models of validation error. Significantly more efficient than grid or random search. Optimal parameters determined by minimizing mean squared error on validation data.

## Key Results

### Model Performance
All three algorithms achieved R-squared values exceeding 0.95 on training data. GBDT with Bayesian optimization showed superior generalization with R-squared of 0.9915 (training) and 0.9076 (testing) for Case 4.

### Generalization to Couette Flow
Models successfully predicted pressure-strain correlation for Couette flow despite being trained only on channel flow data. This demonstrates that models learned fundamental turbulence physics rather than flow-specific patterns.

### Feature Importance
Mean velocity gradient (strain rate) identified as the most influential feature, consistent with physical understanding of pressure-strain correlation mechanisms.

## Requirements

### Core Dependencies
```
python >= 3.8
numpy
pandas
matplotlib
scipy
scikit-learn
xgboost
torch
optuna
jupyter
```

### Data Access
DNS datasets must be downloaded from the Oden Institute Turbulence File Server:
- Channel flow: https://turbulence.oden.utexas.edu/
- Couette flow: https://turbulence.oden.utexas.edu/

Expected data files:
- LM_Channel_XXXX_mean_prof.dat
- LM_Channel_XXXX_vel_fluc_prof.dat
- LM_Channel_XXXX_rey_stress_prof.dat
- Couette flow equivalent files

Place data files in appropriate directory structure as referenced in Phase 1 notebook.

## Usage

### Sequential Execution
Notebooks must be executed in order (Phase 0 through Phase 6) as each phase depends on outputs from previous phases.

```bash
jupyter notebook Phase_0.ipynb
```

### Data Preparation
Phase 1 generates processed datasets saved as CSV files. Subsequent phases load these datasets. If re-running Phase 1, ensure sufficient disk space for intermediate data products.


### Reproducibility
Random seeds are set throughout for reproducibility. However, minor variations may occur due to:
- Hardware differences (CPU vs GPU)
- Floating point arithmetic
- Parallel processing in tree-based methods

## Novel Contributions

This implementation extends the original paper with:

1. **Modern Optimization Techniques:** Bayesian hyperparameter optimization replacing manual tuning, significantly improving model generalization.

2. **Production-Ready Code:** XGBoost implementation providing industry-standard performance and scalability compared to custom implementations.

3. **Comprehensive Documentation:** Detailed explanations of turbulence physics, mathematical formulations, and ML methodology accessible to both CFD and ML practitioners.

4. **Rigorous Validation:** Extensive out-of-distribution testing on Couette flow to verify physics learning rather than pattern memorization.

5. **Open Science:** Complete implementation with all data processing, feature engineering, model training, and evaluation steps fully transparent and reproducible.

## Physical Constraints and Realizability

The pressure-strain correlation models must satisfy:
- Zero trace (energy conservation in incompressible flow)
- Return to isotropy in decaying turbulence
- Rapid Distortion Limit behavior
- Realizability constraints on Reynolds stress anisotropy

Feature importance analysis confirms models learned physically meaningful relationships, with strain rate dominating as expected from turbulence theory.

## Limitations and Future Work

### Current Limitations
- Training limited to wall-bounded shear flows
- Single geometry (plane channel and Couette)
- Incompressible flow assumption
- Limited Reynolds number range

### Recommended Extensions
- Incorporation of multiple flow geometries (pipes, boundary layers, jets)
- Extension to compressible flows
- Physics-informed neural networks incorporating governing equations as constraints
- Integration with CFD solvers for a priori/a posteriori testing
- Uncertainty quantification for model predictions

## Citation

If using this implementation, please cite the original research paper:

```
J.P. Panda and H.V. Warrior (2021)
"Evaluation of Machine Learning Algorithms for Predictive Reynolds Stress Transport Modeling"
arXiv:2105.13641 [physics.flu-dyn]
```

## References

### Primary References
- Lee, M. & Moser, R.D. (2015). Direct numerical simulation of turbulent channel flow up to Re_tau = 5200. Journal of Fluid Mechanics, 774, 395-415.
- Lee, M. & Moser, R.D. (2018). Extreme-scale motions in turbulent plane Couette flows. Journal of Fluid Mechanics, 842, 128-145.
- Speziale, C.G., Sarkar, S. & Gatski, T.B. (1991). Modelling the pressure-strain correlation of turbulence: an invariant dynamical systems approach. Journal of Fluid Mechanics, 227, 245-272.

### Machine Learning for Turbulence
- Duraisamy, K., Iaccarino, G. & Xiao, H. (2019). Turbulence modeling in the age of data. Annual Review of Fluid Mechanics, 51, 357-377.
- Ling, J., Kurzawski, A. & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. Journal of Fluid Mechanics, 807, 155-166.
- Wu, J., Xiao, H. & Paterson, E. (2018). Physics-informed machine learning approach for augmenting turbulence models: A comprehensive framework. Physical Review Fluids, 3(7), 074602.

## License

This implementation is provided for academic and research purposes. Please refer to the original paper for research findings and theoretical contributions.

## Contact

For questions regarding implementation details or research methodology, please open an issue in this repository.

## Acknowledgments

This work implements research conducted at the Department of Ocean Engineering and Naval Architecture, Indian Institute of Technology Kharagpur. DNS data provided courtesy of the Oden Institute for Computational Engineering and Sciences, University of Texas at Austin.
