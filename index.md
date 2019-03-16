---
layout: home
nav_corder: 0
---

# Scala Infer
With the addition of a few keywords, `scala-infer` turns scala into a probabilistic programming language.
To achieve scalability, inference is based on gradients of the (variational approximation to) the
posterior distribution.  Each draw from a distribution is accompanied by a *guide* distribution.

A probabilistic model in `scala-infer` is written as regular scala code.  Values are drawn from
distributions and are used to generate data.  Such a model is known as *generative*, as it provides
an explicit process.

Three new keywords are introduced:
* `infer` to define a model
* `sample` to draw a random variable from a distribution
* `observe` to bind data to distributions in the model

When a value is sampled, two distributions are needed; the *prior* and the (variational
approximation to) the *posterior*.  Actual sample values are drawn from the posterior, but
the prior is the starting point of the buildup of the posterior.  Without observations, the
posterior would be equal to the prior.

Parameters of the variational posterior are optimized by gradient descent.  For each sample from
the model, a backward pass calculates the gradients of the loss function.  For variational inference,
the loss function is the ELBO, a lower bound on the evidence.

Different tactics are used for discrete and continuous variables.  For continuous variables,
the reparametrization trick can be used to obtain a low variance estimator.  Discrete variables
use black-box variational inference, which only requires gradients of the score function to the
parameters.

## Including it in your sbt project
To leverage `scala-infer` in your project, update `plugins.sbt` with
```scala
resolvers += Resolver.bintrayRepo("scala-infer", "maven")
```
and in `build.sbt`, add
```scala
libraryDependencies += "scala-infer" %% "scala-infer" % "0.2"
```

## Including it in a jupyter notebook
With the installation of [Almond](https://almond.sh/) it is possible to use
Scala in a Jupyter environment.  Assuming it is installed, execute
```scala
interp.repositories() ++= Seq(
    coursier.MavenRepository("https://dl.bintray.com/scala-infer/maven")
)
```
followed by
```scala
import $ivy.`scala-infer::scala-infer:0.2`
```
to get the goodness of `scala-infer` at your disposal.
