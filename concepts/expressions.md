---
title: Expressions
nav_order: 1
parent: Concepts
---
# Expressions
The simplest expression is the *Parameter*, which holds a value that is
optimized by the *Optimizer* (typically, the *Adam* optimization algorithm is
used).  It is however possible to do a deterministic transformation of these
parameters before using them as the defining parameters of a distribution.

An example of such a transformation is the *Variational Auto Encoder*.  It
does not define variational parameters for each data point (say, an image).
Instead, a neural network is used to calculate parameters for the guides from
the data point.  In this way, the number of variational parameters is greatly
reduced.  Furthermore, the trained neural network can be used to obtain good
approximations to the posterior distributions of hidden variables for new 
data points.

