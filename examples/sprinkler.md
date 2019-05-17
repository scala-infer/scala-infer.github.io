---
title: Sprinkler system
parent: Examples
nav_order: 1
---

## Example: Sprinkler system
This is the rain-sprinkler-wet-grass system from the Wikipedia entry on [Bayesian
Networks](https://en.wikipedia.org/wiki/Bayesian_network).  It features a number of discrete
(boolean) random variables, an observation (the grass is wet) and an variational posterior
distribution that is optimized to approximate the exact posterior.

```scala
// optimization algorithm: Adam
val sgd = new Adam(alpha = 0.1)

// posterior distribution for the sprinkler, conditional on rain.
// The parameters run over the full real axis, they are mapped to the
// domain [0,1] by the sigmoid transformation.
val inRain = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0))))
val noRain = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0))))

// posterior distribution for the rain
val rainPost = BBVIGuide(Bernoulli(sigmoid(sgd.param(0.0))))

// full model of the rain-sprinkler-grass system.  
val model = infer {

  // conditional sampling of the sprinkler.  The probability that
  // the sprinkler turned on, is dependent on whether it rained or not.
  val sprinkle = {
    rain: Boolean =>
      if (rain) {
        sample(Bernoulli(0.01), inRain)
      } else {
        sample(Bernoulli(0.4), noRain)
      }
  }

  val rain = sample(Bernoulli(0.2), rainPost)
  val sprinkled = sprinkle(rain)

  val p_wet = (rain, sprinkled) match {
    case (true,  true)  => 0.99
    case (false, true)  => 0.9
    case (true,  false) => 0.8
    case (false, false) => 0.001
  }

  // bind model to data / add observation
  observe(Bernoulli(p_wet), true)

  // return quantity we're interested in
  rain
}
```
The example shows a number of features:
* control flow (`if (...) ... else ...`) can be based on random variables
* it's possible to define functions of random variables

