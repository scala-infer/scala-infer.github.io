---
title: Probabilistic Programming
nav_order: 1
---
# Probabilistic Programming
Probabilistic programming languages allow data analysts, scientists and
engineers to model data by describing the process that generated it.  Instead
of manipulating data into a form familiar to an existing model, one can focus
on understanding the data-generating process.  As one works with the data and
the domain that generated it, modelling it effectively means being able to 
iterate on this process.

Existing machine learning toolkits contain out-of-the-box models for regression
and classification.  These *models* are provided because researchers have
worked out ways to efficiently do *inference* on them.  This makes for a 
double-edged sword.  If a problem can be mapped to an existing model, then
it *is* possible to do inference.

If a model *cannot* be mapped to an existing (inference) solution, you'll
suddenly find yourself in a whole different ball-park.  You will need to find
an inference algorithm yourself, for a model that is sufficiently close to the
model you actually want to use.

Deriving custom inference algorithms is certainly possible, but requires
sufficient time to work out the maths alone.  A modest improvement in model
complexity may require a significant increase in the algorithmic complexity for
the inference.  This also translates to tedious, error-prone, implementation
in a programming language.  Practically speaking, this means that model
development can only marginally improve on the out-of-the-box situation.

The probabilistic programming solution to this situation is to move inference
into a separate engine altogether.  Letting the user worry about getting the
model right, the engine will take care of the inference.  This will not be
as efficient as a hand-coded implementation of an analytic solution.  But with
ever more compute power at our disposal, it is plain silly to waste a lot of
time and effort on coming up with an analytic solution to a model and a correct
implementation of the inference algorithm.  For a model which may turn out not
to be an improvement over an existing, simpler, one...

