---
title: Tensors
nav_order: 2
---
## Tensors
While it is possible to treat every data point separately, as demonstrated above, this has some
overhead associated with it.  Often variables and data have some regularity to them, such that
control flow is the same for many data points.  In this case, it is possible to operate on tensor
variables and data.

In general, tensors are multi-dimensional arrays of data.  To deal with these multiple dimensions,
`scala-infer` attaches a type to each dimension.  For instance, when dealing with a batch of data
points:
```scala
case class Batch(size: Int) extends Dim[Batch]

val shape = Batch(2)
val data = Array(0.0f, 1.0f)
val tensor: Tensor[Batch, ArrayTensor] = Tensor(shape, ArrayTensor(shape.sizes, data))
```
where the type of the final `tensor` variable has been added for clarity.  A tensor can be backed
by different data structures.  Above the java native `Array[Float]` is used, but it is also possible
to use Nd4j's `INDArray`'s for example.
