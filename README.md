# Model Generalization

## The question we are asking

Models are useful not because they fit training data, but because they can make predictions about new data. But how do we know whether a model will work on data it has never seen before?

In these two modules, we look at two ways model evaluation can go wrong. First, we explore how a model can overfit its training data and how validation, early stopping, and regularization can help. Then we look at how related examples in biological data can make test performance look better than it really is.

## Modules

### [Overfitting](./overfitting.md)

- Polynomial regression and model complexity
- Underfitting and overfitting
- Training, validation, and test data
- Early stopping
- L2 regularization

### [Leakage in biological data](./leakage.md)

- Sequence families and related examples
- Random splits versus family splits
- How data leakage can produce misleading test scores
- Choosing a split that matches how a model will be used

## Reading

- Whalen, S., Schreiber, J., Noble, W. S., & Pollard, K. S. (2022). Navigating the pitfalls of applying machine learning in genomics. *Nature Reviews Genetics*, *23*, 169–181. https://doi.org/10.1038/s41576-021-00434-9
- Xi, W., & Beer, M. A. (2018). Local epigenomic state cannot discriminate interacting and non-interacting enhancer–promoter pairs with high accuracy. *PLOS Computational Biology*, *14*(12), e1006625. https://doi.org/10.1371/journal.pcbi.1006625
- Stowell, D. (2022). Computational bioacoustics with deep learning: a review and roadmap. *PeerJ*, *10*, e13152. https://doi.org/10.7717/peerj.13152
- Wierucka, K., et al. (2025). Same data, different results? Machine learning approaches in bioacoustics. *Methods in Ecology and Evolution*.
