# p3py
package for PPP-related calculations

## installation

To install the package, call

```bash
pip install .
```

from the command line inside the `p3py` directory.

This package requires the `numpy` package, which will be installed if it is not already installed in the current environment

## Modules

### GLS

The `gls` module contains functions to calculate the GLS estimate and uncertainty. The two functions are `get_gls_estimate()` and `get_gls_unc()`. The function `get_gls_estimate()` calls `get_gls_unc()` and will return both the values and uncertainties for the fitted parameters. 

For example, to solve the "original" PPP problem:

```python
from p3py import get_gls_estimate
import numpy as np

# set up matrices from PPP problem
X = np.array([[1],[1]])
Y = np.array([[1.0],[1.5]])
V = np.array([[0.05,0.06],[0.06,0.1125]])

est, unc = get_gls_estimate(Y, V, X)
est, unc
```

```
(array([[0.88235294]]), array([[0.21828206]]))
```

The function `get_gls_unc()` is available to the user separately because it does not require data points, only a covariance matrix, and can be used to test specific covariance and design matrix combinations without needing a data points column vector.

```python
from p3py import get_gls_unc

unc = get_gls_unc(V, X)
unc
```

```
array([[0.21828206]])
```


### SAMMY

The `sammy` module contains the equations for the Methods 1, 2, 2a, and 2b in the SAMMY manual for 2x2 matrices. 

```python
from p3py.sammy import method1, method2, method2a, method2b

# set up small dn values
r = np.array([10000,12100])
dr = np.array([100,110])
n = 100
dn = 0.5

p, m = method1(r, dr, n, dn, verbose=True)
```

```
109.5023 +/- 0.9205

100.0 +/- 0.5
```

```python
p, m
```
```
(array([[109.50226244],
        [100.        ]]),
 array([[ 0.84727995, -0.27375566],
        [-0.27375566,  0.25      ]]))
```

Each of the functions `method1`, `method2`, `method2a`, and `method2b`, are called the same way.