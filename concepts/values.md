---
title: Values
nav_order: 2
parent: Concepts
---
# Values
A *Value* holds a continuous variable and is able to back-propagate gradients.
During the sampling process, DAGs (directed acyclic graphs) of *Value*s are
built when calculating (log) probabilities of the posterior and the guide.
After the forward pass, the gradient of the error (the difference between
posterior and guide) is back propagated through the *Value*s.

In order to make back propagation efficient, each *Value* should only be
invoked once in the backward pass.  When a *Value* has multiple downstream
uses, however, it would receive multiple gradients.  Aggregation of these
updates occurs in a *Buffer*, which only passes the gradient on when it is
completed.
