---
title: Linear Regression
parent: Examples
nav_order: 2
---
# Linear Regression
Here we showcase linear regression on 2 input variables.  All variables are
continuous here, with some fixed values used to generate a data set and a model
to infer these parameters from the data.
```scala
// generate data; parameters should be recovered by inference algorithm
val data = {
  val alpha = 1.0
  val beta = (1.0, 2.5)
  val sigma = 1.0

  for {_ <- 0 until 100} yield {
    val X = (Random.nextGaussian(), 0.2 * Random.nextGaussian())
    val Y = alpha + X._1 * beta._1 + X._2 * beta._2 + Random.nextGaussian() * sigma
    (X, Y)
  }
}

// set up variational approximation to the posterior distribution
val aPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0)))))
val b1Post = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
val b2Post = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))
val errPost = ReparamGuide(Normal(Param(0.0), exp(Param(0.0))))

// Draw variables from prior distributions and link those
// variables to the posterior approximation.
val model = infer {
  val a = sample(Normal(0.0, 1.0), aPost)
  val b1 = sample(Normal(0.0, 1.0), b1Post)
  val b2 = sample(Normal(0.0, 1.0), b2Post)
  val err = exp(sample(Normal(0.0, 1.0), errPost))

  // iterate over data points to define the observations
  data.foreach[Unit] {
    case ((x1, x2), y) =>
      observe(Normal(a + b1 * x1 + b2 * x2, err), y: Real)
  }

  // return the values that we're interested in
  (a, b1, b2, err)
}

// choose an optimization algorithm
// each parameter could have its own optimizer
val adam = new Adam(alpha = 0.1)
val interpreter = new OptimizingInterpreter(adam)

// warm up
// each sample of the model triggers a gradient descent step
Range(0, 1000).foreach { i =>
  interpreter.reset()
  model.sample(interpreter)
}

// print some samples
Range(0, 10).foreach { i =>
  interpreter.reset()
  val l = model.sample(interpreter)
  val values = (l._1.v, l._2.v, l._3.v, l._4.v)
  println(s"  $values")
}
```
Here, we not only inject the variational posterior distribution into the model,
but the data as
well.  Some things to note here 
* we can naturally iterate over the data and declare observations - the used
  data types `Seq` and `Tuple2` have no special meaning and neither has the
  `foreach` method
* while real parameters and random variables run over the whole real axis, they
  can be mapped to the interval `(0, Inf)` by the `exp` function

