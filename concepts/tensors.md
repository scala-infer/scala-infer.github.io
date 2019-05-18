---
title: Tensors
nav_order: 4
parent: Concepts
---
# Tensors
While it is possible to treat every data point separately, this has some
overhead associated with it.  Often variables and data have some regularity to
them, such that control flow is the same for many data points.  In this case,
it is possible to operate on tensor variables and data.

Tensors are multi-dimensional arrays of data.  To deal with these multiple
dimensions, `scala-infer` attaches a type to each dimension.  For instance,
when dealing with a batch of data points:
```scala
case class Batch(size: Int) extends Dim[Batch]

val shape = Batch(2)
val data = Array(0.0f, 1.0f)
val tensor = Value(ArrayTensor(shape.sizes, data), shape)
```
A tensor can be backed by different data structures.  Above the java native
`Array[Float]` is used, but it is also possible to use Nd4j's `INDArray`'s for
example.

Higher-order tensors are obtained by concatenating dimensions into a *shape*.
```scala
case class Width(size: Int) extends Dim[Width]
val width = Width(3)

case class Height(size: Int) extends Dim[Height]
val height = Height(3)

type GridShape = Width :#: Height
val gridShape = width :#: height
```

## Tensordot
While for tensors element-wise operations are useful and needed in many models,
an important operation that does not exist for scalar variables is *tensordot*,
AKA *tensor contraction*.  A contraction of two tensors multiplies their
elements, summing over their common dimensions.

This generalizes well-known multi-dimensional operations like
* vector inner product
* matrix-vector product
* matrix-matrix product

```scala
case class A(size: Int) extends Dim[A]
case class B(size: Int) extends Dim[B]
case class C(size: Int) extends Dim[C]

val a = A(1)
val b = B(2)
val c = C(3)

val xShape = a :#: b
val xData = ArrayTensor(xShape.sizes, Array(1f, 2f))

val yShape = b :#: c
val yData = ArrayTensor(yShape.sizes, Array(1f, 2f, 3f, 4f, 5f, 6f))

val x = Value(xData, xShape)
val y = Value(yData, yShape)

// inner product of vector with itself -> returns scalar
val xx: Value[ArrayTensor, Scalar] = x :*: x

// matrix matrix product - sum over common dimension
val z: Value[ArrayTensor, C :#: A] = x :*: y
```
where the types of `xx` and `z` have been added for clarity (scalas inference
mechanism is perfectly able to infer this by itself, thank you very much).
