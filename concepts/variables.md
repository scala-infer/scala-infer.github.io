---
title: Variables
nav_order: 3
parent: Concepts
---
# Variable
When a sample is drawn from a distribution, `scala-infer` creates a *Variable*
behind the scenes.  It keeps track of the dependencies between these samples,
to make it possible to calculate the *Markov Blanket*.  When approximating the
posterior distribution for a discrete variable, this blanket contains all other
random variables whose value affects the (true) posterior.  Limiting the
approximating procedure to just the probabilities in the blanket (rather than
the complete posterior) eliminates noise from other parts of the model.

As *Variable*s contain the dependencies between random variables, they
effectively build up the (dynamic) Bayesian Network for a sample from the
model.  Armed with this information, they perform a crucial role in the
backward phase of back-propagation.  As back-propagation is effectively
running the model in reverse and *Variable*s know the (dependencies in the)
execution graph, they take care of the "flushing" of gradients in the
*Buffer*s.

Since the bookkeeping that goes along with the use of *Variable*s is quite
tedious and error-prone, they are hidden from end-users of `scala-infer`.
The `infer` macro rewrites code such that the bookkeeping is performed -
this is indeed the primary job of the macro.
