---
title: Guides
nav_order: 3
---
# Guides
Guides inject the approximation to the posterior into the model definition.  When the (variational)
inference algorithm runs, the difference between the approximation and the exact posterior is
minimized.  When sampling from the model, variables are sampled from the (distributions in the) guide.

## Reparametrization Gradient
Continuous random variables from suitable distributions can be recast in a "reparametrized" form.
A sample is obtained by sampling a fixed-parameter distribution, followed by a deterministic
transformation specified by the parameters of the target distribution.

The variance of the gradient obtained for the parameters is reduced further by using the "Path
Derivative", as detailed in [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for
Variational Inference](https://arxiv.org/abs/1703.09194) by Roeder, Wu and Duvenaud.

## Black-Box Variational Inference
Dealing with discrete random variables is necessary to support a general programming language,
with control flow depending on sampled variables.  For discrete variables reparametrization is
not possible and we resort to Black-Box Variational Inference.  This suffers from large variance
on estimates of the gradient of the posterior probability with respect to the variational
parameters.

Two ways of reducing the variance are
* Rao-Blackwellization
* Control variates

The first (Rao-Blackwellization) is implemented by limiting, for each variable, the posterior to
elements in its Markov blanket.  The prior probability links the variable to its parents,
likelihoods of observations and prior probabilities of downstream variables link it to children
and their (other) parents.  This procedure eliminates irrelevant other variables as sources of
variance.

A simple control variate (moving average of the log-posterior) is used to reduce variance of
the gradient further.

For further reading, see [Black Box Variational Inference](https://arxiv.org/abs/1401.0118) by
Ranganath, Gerrish and Blei.

