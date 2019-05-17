---
title: Mixture
parent: Examples
nav_order: 3
---
## Example: Two component Mixture
So far, we've seen examples of global variables being fit to a variational posterior.  However, it's
also possible to fit local variables.  Focussing on the model definition part:
```scala
val data: Seq[Double] = ???

val dataWithGuides = data.map { datum =>
  (datum, BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0)))))
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
Here, we create a variational parameter for each data point - corresponding to the probability that
the data point belongs to the first cluster.

