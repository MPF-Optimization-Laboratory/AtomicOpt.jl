# Polar Calculus

The operations for gauge and support functions are governed by a simple calculus. The following atoms are defined:

- `support(A, z)`: gives the support value ``\sigma_{\mathcal A}(z)``. 
- `expose(A, z)`: gives an atom ``a\in\mathcal{A}`` such that ``\langle a, z\rangle = \sigma_{\mathcal A}(z)``.
- `length(A)`: the dimension of elements in ``\mathcal A``.

For each atom in ``a\in\mathcal A``, the following methods are implemented:
- `M*a`: the product of a linear operator ``M`` and ``a`` as a vector of `size(M,1)`.
- `vec(a)`: the atom in the natural space of ``A``.
- `length(a)`: the dimension of the atom ``a``.

## One-norm ball

```@docs
OneBall
```

## Sum of atomic sets

Suppose that ``\mathcal{A}=\mathcal{A_1} + \mathcal{A_2}``. The support function for ``\mathcal{A}`` is seperable, i.e.,
```math
\sigma_{\mathcal{A}}(z) =  \sigma_{\mathcal{A_1}}(z) + \sigma_{\mathcal{A_2}}(z).
```
This rule is implemented in the following method.
```@docs
support(::SumAtomicSet, ::Vector)
```

## Scaling atomic sets

Suppose that we have a method for computing the gauge and support-function
operations associated with an atomic set ``\mathcal{A}``. Then the gauge and
support-function operations for the transformed atomic set

```math
M\mathcal{A}=\{ Ma \mid a\in\mathcal{A}\},
```

where ``M`` is an invertible linear map, can be obtained from the corresponding operations involving ``\mathcal{A}`` alone. These operations are supported for the `ScaledAtomicSet` type.

### Gauge to a scaled atomic set

From the definition of a gauge,


```math
\begin{aligned}
\gamma(x \mid {M\mathcal{A}})
   &= \inf\{ \lambda\ge0 \mid x\in \lambda M\mathcal{A}\} 
\\ &= \inf\{ \lambda\ge0 \mid M^{-1}x\in\lambda\mathcal{A}\}
\\ &= \gamma(M^{-1}x \mid \mathcal{A}).
\end{aligned}
```


