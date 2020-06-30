# Pairwise Permutations Algorithm

This package implements the pairwise permutations algorithm, which is based on permutation importance and can be used to rank features in your model.

In this small repository, you will find the implementation of the PPA algorithm, as well as a sample example file using the package. The package is agnostic in regards to model object as well as the objective function. However, the model object implement the *predict* and the *set_params* methods and the objective function must follow the following syntax:

```
metric_func(y_test, y_pred)
```

