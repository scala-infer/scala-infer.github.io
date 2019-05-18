---
title: Empirical Bayes
parent: Examples
nav_order: 5
---
# Empirical Bayes
The Bayesian approach to inference expects the practitioner to carefully
consider their prior beliefs and encode those in prior distributions.  In
practice, however, not enough domain knowledge may be available when
modelling to do this.

Especially when there are more features available than there are points
in the dataset, pruning the features becomes important.  This is essentially
Occams Razor - when two models can explain the data, the simpler model is
preferred.  The bayesian definition of "simple" is to have a high *marginal
likelihood*.  This balances the bias that is introduced by reducing the number
of parameters with the reduction in variance that goes along with it.

## Generating the data
Let's create a dataset with 2 explanatory variables, `a` and `b`, and one
dependent variable, `y`.  The variable `a` indeed influences `y`, while `b`
is independent.
```scala
case class Record(a: Float, b: Float, y: Float)

// Tensor shape - let's make it typed!
case class Batch(size: Int) extends Dim[Batch]
val batch = Batch(1000)

val (a_vals, b_vals, y_vals) = {
    val a_weight = 1.0
    val b_weight = 0.0
    val noise = 0.5
    
    val data = for { _ <- 0 until batch.size } yield {
        val a = Random.nextGaussian()
        val b = Random.nextGaussian()
        val y = a_weight * a + noise
        Record(a.toFloat, b.toFloat, y.toFloat)
    }

    (
        Value(ArrayTensor(batch.sizes, data.map { _.a }.toArray), batch),
        Value(ArrayTensor(batch.sizes, data.map { _.b }.toArray), batch),
        Value(ArrayTensor(batch.sizes, data.map { _.y }.toArray), batch)
    )
}
```

## Create the model
In the model, we now not just introduce parameters for the variational
approximation (`a_post_mu`, `a_post_s`, `b_post_mu` and `b_post_s`).  We also
include parameters for the *prior* distributions, `a_prior_s` and `b_prior_s`.

Note that we here treat `a` and `b` identically, as we want the optimization
procedure to figure out which parameters are relevant by itself.
```scala
val a_prior_s = Param(0.0)
val b_prior_s = Param(0.0)

val a_post_mu = Param(0.0)
val a_post_s = Param(0.0)
val a_guide = ReparamGuide(Normal(a_post_mu, exp(a_post_s)))

val b_post_mu = Param(0.0)
val b_post_s = Param(0.0)
val b_guide = ReparamGuide(Normal(b_post_mu, exp(b_post_s)))

val noise_mu = Param(0.0)
val noise_s = Param(0.0)
val noise_guide = ReparamGuide(Normal(noise_mu, exp(noise_s)))

val model = infer {
    val a_weight = sample(Normal(0.0, exp(a_prior_s)), a_guide)
    val b_weight = sample(Normal(0.0, exp(b_prior_s)), b_guide)
    val noise = sample(Normal(0.0, 1.0), noise_guide)

    observe(Normal(
        broadcast(a_weight, batch) * a_vals
        + broadcast(b_weight, batch) * b_vals,
        broadcast[Batch, ArrayTensor](exp(noise), batch)
    ), y_vals)
}
```

## Running the optimization
After optimization, we can see a clear difference between relevant and
irrelevant parameters.  Parameter `a_prior_s` drives the standard deviation
in the prior for `a_weight` to `1`, while the same for `b_weight` gets close to
`0`.

## Notebook
The Jupyter notebook with the code is available at
[Automatic Relevance Determination.ipynb](https://github.com/scala-infer/notebooks/blob/master/Automatic Relevance Determination.ipynb)
in the [scala-infer notebooks](https://github.com/scala-infer/notebooks) project.
