---
title: Mixture
parent: Examples
nav_order: 3
---
## Example: Two component Mixture
So far, we've seen examples of global variables being fit to a variational
posterior.  However, it's also possible to fit local variables.  Focussing on
the model definition part:
```scala
val data: Seq[Double] = ???

val dataWithGuides = data.map { datum =>
  (datum, BBVIGuide(Bernoulli(sigmoid(Param(0.0)))))
}

val model = infer {
  val p = sigmoid(sample(Normal(0.0, 1.0), pPost))
  val mu1 = sample(Normal(0.0, 1.0), mu1Post)
  val mu2 = sample(Normal(0.0, 1.0), mu2Post)
  val sigma = exp(sample(Normal(0.0, 1.0), sigmaPost))

  dataWithGuides.foreach[Unit] {
    case (value, guide) =>
      if (sample(Bernoulli(p), guide)) {
        observe(Normal(mu1, sigma), value: Real)
      } else {
        observe(Normal(mu2, sigma), value: Real)
      }
  }

  (p, mu1, mu2, sigma)
}
```
Here, we create a variational parameter for each data point - corresponding to
the probability that the data point belongs to the first cluster.

## Amortized Inference
When creating a separate parameter for each data point, we are effectively finding
points on a function from the `value` domain to the `p` domain.  An alternative
way of specifying this function is by making an Ansatz for the function and fit
its parameters.
```scala
val intercept = Param(0.0)
val slope = Param(1.0)

val dataWithDist = data.map { datum =>
    val local = intercept + slope * datum
    (datum, BBVIGuide(Bernoulli(sigmoid(local))))
}
```

The upshot of this approach is that the function can be used to obtain the
posterior distribution immediately for new data points.  No optimization of new
parameters is needed.

## Notebook
The notebook with the code is available at [https://github.com/scala-infer/notebooks/blob/master/Mixture.ipynb](https://github.com/scala-infer/notebooks/blob/master/Mixture.ipynb).
