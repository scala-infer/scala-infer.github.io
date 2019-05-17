---
title: Concepts
nav_order: 2
has_children: true
---
# Concepts
Scala-infer has a few organising principles, which determine what type of code
should go where.  Central to the design is the sampling procedure of
variational inference.  The model is executed (sampled) repeatedly, with the
approximation to the posterior getting better with each sample.

Inside the model definition, variables obtain a *Value* which contains an actual
sample (or a value derived from samples).  Outside the model definition, only
symbolic manipulation of *Expression*s is possible.  E.g. when defining a
parameter for a *Guide* or a prior distribution, this is an *Expression*.

To sample from the model, an *Interpreter* is used to evaluate an *Expression*
into a *Value*.  This same interpreter can also be used outside the model to
get values for parameters.
