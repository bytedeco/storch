/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package ops

import torch.internal.NativeConverters
import NativeConverters.*
import Layout.Strided
import Device.CPU
import MemoryFormat.Contiguous
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BoolOptional,
  ByteVector,
  GenericDict,
  IValue,
  LongOptional,
  MemoryFormatOptional,
  OutputArchive,
  Scalar,
  ScalarTypeOptional
}
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch.ScalarType
import torch.nn.modules.Module as ModelModule
import torch.numpy.matrix.NDArray
import java.nio.file.{Files, Path, Paths}
import scala.collection.immutable.{SeqMap, VectorMap}
import torch.internal.NativeConverters.fromNative

import scala.reflect.ClassTag

/** Creation Ops
  *
  * https://pytorch.org/docs/stable/torch.html#creation-ops
  */
private[torch] trait CreationOps {

// TODO sparse_coo_tensor
// TODO as_tensor
// TODO as_strided
// TODO frombuffer

  def as_tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean = false
  ): Tensor[ScalaToDType[U]] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def tensors[U <: ScalaType: ClassTag, D <: DType | Derive](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      dtype: D = derive // torch.float32
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] = {

    val derivedDType = dtype match
//      case _: Derive => scalaToDType(fillValue)
      case t: DType => t
    val tensor = Tensor.apply(data, layout, device, requires_grad).to(dtype = derivedDType)
    tensor.requires_grad_(requires_grad)
    fromNative(tensor.native)
  }

  def tensor[U <: ScalaType: ClassTag, D <: DType](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      dtype: D = torch.float32
  ): Tensor[D] = {
    val tensor = Tensor.apply(data, layout, device, requires_grad)
    tensor.requires_grad_(requires_grad)
    fromNative(tensor.native).to(dtype = dtype)
  }

  def tensorz[U <: ScalaType: ClassTag, D <: DType](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean = false,
      layout: Layout = Strided,
      device: Device = CPU
  ): Tensor[ScalaToDType[U]] = {
    val tensor = Tensor.apply(data, layout, device, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean
  ): Tensor[ScalaToDType[U]] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  //    dtype match {
  //      case t: D => tensor.to(dtype = t)
  //      case t: Option[D] => if (t.isDefined) tensor.to(dtype = t.get) else tensor
  //      case None => tensor
  //    }
//  def tensor[U <: ScalaType: ClassTag](
//      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
//        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
//      requires_grad: Boolean
//  ): Tensor[ScalaToDType[U]] = Tensor.apply(data, requires_grad)

  def tensor[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean,
      device: Device
  ): Tensor[ScalaToDType[U]] = {
    val tensor = Tensor.apply(data, requires_grad, device)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def intTensor(
      data: Int | Seq[Int] | Seq[Seq[Int]] | Seq[Seq[Seq[Int]]] | Seq[Seq[Seq[Seq[Int]]]] |
        Seq[Seq[Seq[Seq[Seq[Int]]]]],
      requires_grad: Boolean = false
  ): Tensor[Int32] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def floatTensor(
      data: Float | Seq[Float] | Seq[Seq[Float]] | Seq[Seq[Seq[Float]]] |
        Seq[Seq[Seq[Seq[Float]]]] | Seq[Seq[Seq[Seq[Seq[Float]]]]],
      requires_grad: Boolean = false
  ): Tensor[Float32] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def bfloat16Tensor(
      data: Float | Seq[Float] | Seq[Seq[Float]] | Seq[Seq[Seq[Float]]] |
        Seq[Seq[Seq[Seq[Float]]]] | Seq[Seq[Seq[Seq[Seq[Float]]]]],
      requires_grad: Boolean = false
  ): Tensor[BFloat16] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor.to(dtype = torch.bfloat16)
  }

  def boolTensor(
      data: Boolean | Seq[Boolean] | Seq[Seq[Boolean]] | Seq[Seq[Seq[Boolean]]] |
        Seq[Seq[Seq[Seq[Boolean]]]] | Seq[Seq[Seq[Seq[Seq[Boolean]]]]],
      requires_grad: Boolean = false
  ): Tensor[Bool] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def byteTensor(
      data: Byte | Seq[Byte] | Seq[Seq[Byte]] | Seq[Seq[Seq[Byte]]] | Seq[Seq[Seq[Seq[Byte]]]] |
        Seq[Seq[Seq[Seq[Seq[Byte]]]]],
      requires_grad: Boolean = false
  ): Tensor[Int8] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def shortTensor(
      data: Short | Seq[Short] | Seq[Seq[Short]] | Seq[Seq[Seq[Short]]] |
        Seq[Seq[Seq[Seq[Short]]]] | Seq[Seq[Seq[Seq[Seq[Short]]]]],
      requires_grad: Boolean = false
  ): Tensor[Int16] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def longTensor(
      data: Long | Seq[Long] | Seq[Seq[Long]] | Seq[Seq[Seq[Long]]] | Seq[Seq[Seq[Seq[Long]]]] |
        Seq[Seq[Seq[Seq[Seq[Long]]]]],
      requires_grad: Boolean = false
  ): Tensor[Int64] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def doubleTensor(
      data: Double | Seq[Double] | Seq[Seq[Double]] | Seq[Seq[Seq[Double]]] |
        Seq[Seq[Seq[Seq[Double]]]] | Seq[Seq[Seq[Seq[Seq[Double]]]]],
      requires_grad: Boolean = false
  ): Tensor[Float64] = {
    val tensor = Tensor.apply(data, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  def as_strided[D <: DType](
      input: Tensor[D],
      size: Seq[Int],
      stride: Seq[Int],
      storageOffset: Option[Int] = None
  ): Tensor[D] = {
    if storageOffset.isDefined then
      fromNative[D](
        torchNative.as_strided(
          input.native,
          size.map(_.toLong).toArray,
          stride.map(_.toLong).toArray,
          new LongOptional(storageOffset.get.toLong)
        )
      )
    else
      fromNative[D](
        torchNative.as_strided(
          input.native,
          size.map(_.toLong).toArray,
          stride.map(_.toLong).toArray,
          new LongOptional()
        )
      )
  }

  def as_strided[D <: DType](input: Tensor[D], size: Seq[Int], stride: Seq[Int]): Tensor[D] = {
    fromNative[D](
      torchNative.as_strided(input.native, size.map(_.toLong).toArray, stride.map(_.toLong)*)
    )
  }
// def zeros[D <: DType](size: Int*): Tensor[Float32] =
//   zeros[D](size.toSeq)
  def cumsum[D <: DType](input: Tensor[D], dim: Int, dtype: D = float32): Tensor[D] =
    fromNative(
      torchNative.cumsum(input.native, dim.toLong, ScalarTypeOptional(dtype.toScalarType))
    )

  /** Returns a tensor filled with the scalar value `0`, with the shape defined by the variable
    * argument `size`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @tparam T
    * @return
    *
    * @group creation_ops
    */
  def zeros_raw[D <: DType](size: Int*)(using dtype: D = float32)(using
      requires_grad: Boolean = false
  ): Tensor[D] = this.zeros(size, dtype, Strided, CPU, requires_grad)

  def ones_raw[D <: DType](size: Int*)(using dtype: D = float32)(using
      requires_grad: Boolean = false
  ): Tensor[D] = this.ones(size, dtype, Strided, CPU, requires_grad)

  def zeros_raw[D <: DType](
      size: Seq[Int] | Int,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.zeros(size, dtype, Strided, CPU, requires_grad)

  def ones_raw[D <: DType](
      size: Seq[Int] | Int,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.ones(size, dtype, Strided, CPU, requires_grad)

  def zeross[D <: DType](
      size: Int*
  )(implicit dtype: D = float32, requires_grad: Boolean = false): Tensor[D] =
    this.zeros(size = size, dtype = dtype, layout = Strided, device = CPU, requires_grad = false)

  def oness[D <: DType](
      size: Int*
  )(implicit dtype: D = float32, requires_grad: Boolean = false): Tensor[D] =
    this.ones(size = size, dtype = dtype, layout = Strided, device = CPU, requires_grad = false)

  def emptys[D <: DType](
      size: Int*
  )(implicit dtype: D = float32, requires_grad: Boolean = false): Tensor[D] = this.empty(
    size = size,
    dtype = dtype,
    layout = Strided,
    device = CPU,
    requires_grad = false,
    pinMemory = false,
    memoryFormat = Contiguous
  )

  //  def ones[D <: DType](
//                         size: Seq[Int] | Int,
//                         requires_grad: Boolean
//                       ): Tensor[D] = this.ones(size, torch.float32, Strided, CPU, requires_grad)
//
  def zeros[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def zeros[D <: DType](
      size: Int*
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def zeros[D <: DType](
      s1: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1).map(_.toLong).toArray
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def zeros[D <: DType](
      s1: Int,
      s2: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2).map(_.toLong).toArray
    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
  def zeros[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2, s3).map(_.toLong).toArray

    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def zeros[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2, s3, s4).map(_.toLong).toArray

    fromNative(
      torchNative.torch_zeros(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def zeros_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_zeros_like)

  /** @group creation_ops
    */
  def zerosLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_zeros_like)

  /** Returns a tensor filled with the scalar value `1`, with the shape defined by the variable
    * argument `size`.
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    *
    * @group creation_ops
    */
  def ones[D <: DType](
      size: Seq[Int] | Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray
      case s: Int      => Array(s.toLong)
    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  def ones[D <: DType](
      s1: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1).map(_.toLong).toArray

    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
  def ones[D <: DType](
      s1: Int,
      s2: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2).map(_.toLong).toArray

    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def ones[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2, s3).map(_.toLong).toArray

    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )

  def ones[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int
  ): Tensor[D] =
    val nativeSize = Seq(s1, s2, s3, s4).map(_.toLong).toArray

    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
  def ones[D <: DType](
      size: Int*
  ): Tensor[D] =
    val nativeSize = size match
      case s: Seq[Int] => s.map(_.toLong).toArray

    fromNative(
      torchNative.torch_ones(
        nativeSize,
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
  def ones_like[D1 <: DType](t1: Tensor[D1]): Tensor[D1] =
    fromNative(torchNative.ones_like(t1.native))

  def ones_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_ones_like)

  /** @group creation_ops */
  def onesLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_ones_like)

// format: off
/** Returns a 1-D tensor of size $`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`$ with values
  * from the interval ``[start, end)`` taken with common difference `step` beginning from `start`.
  *
  * Note that non-integer `step` is subject to floating point rounding errors when comparing against `end`;
  * to avoid inconsistency, we advise adding a small epsilon to `end` in such cases.
  *
  * $$
  * \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
  * $$
  *
  * @param start
  *   The starting value for the set of points. Default: ``0``.
  * @param end
  *   The ending value for the set of points
  * @param step
  *   The gap between each pair of adjacent points. Default: ``1``.
  *   
  * @group creation_ops
  */
// format: on

  def arange[D <: DType, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      end: End
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] = {
    fromNative(
      torchNative.torch_arange(
        toScalar(0),
        toScalar(end),
        toScalar(1),
        NativeConverters.tensorOptions(int32, Strided, CPU, false)
      )
    )
  }

  def arange[D <: DType, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      end: End,
      step: Step
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] = {
    fromNative(
      torchNative.torch_arange(
        toScalar(0),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(float32, Strided, CPU, false)
      )
    )
  }

  def arange[
      D <: BFloat16 | FloatNN: Default,
      Start <: ScalaType,
      End <: ScalaType,
      Step <: ScalaType
  ](
      end: End,
      step: Step,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = {
    fromNative(
      torchNative.torch_arange(
        toScalar(0),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(float32, Strided, CPU, requires_grad)
      )
    ).to(dtype = implicitly[Default[D]].dtype)
  }

  def arange[D <: DType | Derive, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      start: Start = 0,
      end: End,
      step: Step = 1,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] =
    val derivedDType = dtype match
      case _: Derive => derivedArangeType(start, end, step)
      case t: DType  => t
    fromNative(
      torchNative.torch_arange(
        toScalar(start),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad)
      )
    )

  def range[D <: DType | Derive, Start <: ScalaType, End <: ScalaType, Step <: ScalaType](
      start: Start = 0,
      end: End,
      step: Step = 1,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[DTypeOrDeriveArange[D, Start, End, Step]] =
    val derivedDType = dtype match
      case _: Derive => derivedArangeType(start, end, step)
      case t: DType  => t
    fromNative(
      torchNative.torch_range(
        toScalar(start),
        toScalar(end),
        toScalar(step),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad)
      )
    )

  /** @group creation_ops */
  def linspace[D <: DType](
      start: Double,
      end: Double,
      steps: Int,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] =
    fromNative(
      torchNative.torch_linspace(
        new Scalar(start),
        new Scalar(end),
        steps.toLong,
        NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
      )
    )

  /** @group creation_ops */
  def logspace[D <: DType](
      start: Double,
      end: Float,
      steps: Int,
      base: Double = 10.0,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ) = fromNative(
    torchNative.torch_logspace(
      new Scalar(start),
      new Scalar(end),
      steps.toLong,
      base,
      NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
    )
  )

  /** Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    *
    * @param n
    *   the number of rows
    * @param m
    *   the number of columns with default being `n`
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requires_grad
    *   If autograd should record operations on the returned tensor.
    *
    * @group creation_ops
    */
  def eye[D <: DType](
      n: Int,
      m: Option[Int] | Int = None,
      dtype: D = float32,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[D] = m match
    case None =>
      fromNative(
        torchNative.torch_eye(
          n,
          NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
        )
      )
    case Some(m) =>
      fromNative(
        torchNative.torch_eye(
          n,
          m,
          NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
        )
      )
    case m: Int =>
      fromNative(
        torchNative.torch_eye(
          n,
          m.toLong,
          NativeConverters.tensorOptions(dtype, layout, device, requires_grad)
        )
      )

// def empty(size: Long*): Tensor[D] = fromNative(torchNative.torch_empty(size*))

  /** Returns a tensor filled with uninitialized data.
    *
    * @group creation_ops
    */
  def empty_raw[D <: DType](
      size: Seq[Int],
      dtype: D,
      requires_grad: Boolean
  ): Tensor[D] = this.empty(size, dtype, Strided, CPU, requires_grad, false, Contiguous)

  def empty_raws[D <: DType](size: Int*)(using dtype: D = float32)(using
      requires_grad: Boolean = false
  ): Tensor[D] = this.empty(size, dtype, Strided, CPU, requires_grad, false, Contiguous)

  def empty[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int
  ): Tensor[D] =
    val size = Seq(s1, s2, s3, s4)
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(float32, Strided, CPU, false)
          .pinned_memory(BoolOptional(false)),
        new MemoryFormatOptional(Contiguous.toNative)
      )
    )
  def empty[D <: DType](
      s1: Int,
      s2: Int,
      s3: Int
  ): Tensor[D] =
    val size = Seq(s1, s2, s3)
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(float32, Strided, CPU, false)
          .pinned_memory(BoolOptional(false)),
        new MemoryFormatOptional(Contiguous.toNative)
      )
    )

  def empty[D <: DType](
      s1: Int,
      s2: Int
  ): Tensor[D] =
    val size = Seq(s1, s2)
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(float32, Strided, CPU, false)
          .pinned_memory(BoolOptional(false)),
        new MemoryFormatOptional(Contiguous.toNative)
      )
    )
  def empty[D <: DType](
      size: Int*
  ): Tensor[D] =
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(float32, Strided, CPU, false)
          .pinned_memory(BoolOptional(false)),
        new MemoryFormatOptional(Contiguous.toNative)
      )
    )
  def empty[D <: DType](
      size: Seq[Int],
      dtype: D,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean,
      pinMemory: Boolean = false,
      memoryFormat: MemoryFormat = Contiguous
  ): Tensor[D] =
    fromNative(
      torchNative.torch_empty(
        size.toArray.map(_.toLong),
        NativeConverters
          .tensorOptions(dtype, layout, device, requires_grad)
          .pinned_memory(BoolOptional(pinMemory)),
        new MemoryFormatOptional(memoryFormat.toNative)
      )
    )

  /** Returns an uninitialized tensor with the same size as input.
    *
    * `torch.empty_like(input)` is equivalent to `torch.empty(input.size(), dtype=input.dtype,
    * layout=input.layout, device=input.device`).
    *
    * @group creation_ops
    */
  def emptyLike[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_empty_like)

  def empty_like[D <: DType, D2 <: DType | Derive](
      input: Tensor[D],
      dtype: D2 = derive,
      layout: Layout | Derive = derive,
      device: Device | Derive = derive,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    xLike(input, dtype, layout, device, requires_grad, memoryFormat, torchNative.torch_empty_like)

  // // TODO emptyStrided

  /** Creates a tensor of size `size` filled with `fillValue`. The tensor's dtype is inferred from
    * `fillValue`.
    *
    * @param size
    *   a sequence of integers defining the shape of the output tensor.
    * @param fillValue
    *   the value to fill the output tensor with.
    * @param dtype
    *   the desired data type of the returned tensor.
    * @param layout
    *   the desired layout of the returned Tensor.
    * @param device
    *   the desired device of the returned tensor.
    * @param requires_grad
    *   If autograd should record operations on the returned tensor.
    * @tparam T
    *   the data type of the returned tensor, or `Default` if the type should be derived from
    *   `fillValue`.
    * @tparam U
    *   the data type of `fillValue`.
    * @return
    *   the newly created tensor.
    *
    * @group creation_ops
    */
  def full[D <: DType | Derive, U <: ScalaType](
      size: Seq[Int],
      fillValue: U,
      dtype: D = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val derivedDType = dtype match
      case _: Derive => scalaToDType(fillValue)
      case t: DType  => t
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad)
      )
    )

  def full[D <: DType, U <: ScalaType](
      s1: Int,
      s2: Int,
      fillValue: U,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val size = Seq(s1, s2)
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )

  def full[D <: DType, U <: ScalaType](
      s1: Int,
      s2: Int,
      s3: Int,
      fillValue: U,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =

    val size = Seq(s1, s2, s3)
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )

  def full[D <: DType, U <: ScalaType](
      s1: Int,
      s2: Int,
      s3: Int,
      s4: Int,
      fillValue: U,
      dtype: D,
      requires_grad: Boolean
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =

    val size = Seq(s1, s2, s3, s4)
    fromNative(
      torchNative.torch_full(
        size.toArray.map(_.toLong),
        toScalar(fillValue),
        NativeConverters.tensorOptions(dtype, Strided, CPU, requires_grad)
      )
    )
    // input (Tensor) – float tensor to quantize
    // scales (Tensor) – float 1D tensor of scales to use, size should match input.size(axis)
    // zero_points (int) – integer 1D tensor of offset to use, size should match input.size(axis)
    // axis (int) – dimension on which apply per-channel quantization
    // dtype (torch.dtype) – the desired data type of returned tensor. Has to be one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32
  // torch.quantize_per_channel(input, scales, zero_points, axis, dtype) → Tensor

  // dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_channel[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scales: Tensor[Float32],
      zero_points: Tensor[Int64],
      axis: Int,
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(
      torchNative.quantize_per_channel(
        input.native,
        scales.native,
        zero_points.native,
        axis.toLong,
        dtype
      )
    )
  }
//  torch.quantize_per_tensor(input, scale, zero_point, dtype) → Tensor
  //    public static native Tensor quantize_per_tensor(@Const @ByRef Tensor var0, double var1, @Cast({"int64_t"}) long var3, ScalarType var5);
//input (Tensor) – float tensor or list of tensors to quantize
//
//scale (float or Tensor) – scale to apply in quantization formula
//
//zero_point (int or Tensor) – offset in integer value that maps to float zero
//
//dtype (torch.dtype) – the desired data type of returned tensor. Has to be one of the quantized dtypes: torch.quint8, torch.qint8, torch.qint32

  //// dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_tensor[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scales: Tensor[Float32],
      zero_points: Tensor[Int64],
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(
      torchNative.quantize_per_tensor(input.native, scales.native, zero_points.native, dtype)
    )
  }

// //dtypes: torch.quint8, torch.qint8, torch.qint32
  def quantize_per_tensor_single[D <: DType, U <: ScalaType](
      input: Tensor[D],
      scale: Double,
      axis: Int,
      dtype: ScalarType = ScalarType.QUInt8
  ): Tensor[?] = {

    fromNative(torchNative.quantize_per_tensor(input.native, scale, axis.toLong, dtype))
  }

  def full_like[D <: DType, D1 <: Derive, U <: ScalaType](
      input: Tensor[D],
      fillValue: U,
      dtype: D1 = derive,
      layout: Layout = Strided,
      device: Device = CPU,
      requires_grad: Boolean = false,
      memoryFormat: MemoryFormat = MemoryFormat.Preserve
  ): Tensor[DTypeOrDeriveFromScalar[D, U]] =
    val derivedDType = dtype match
      case _: Derive => scalaToDType(fillValue)
//      case t: DType => t
    fromNative(
      torchNative.torch_full_like(
        input.native,
        toScalar(fillValue),
        NativeConverters.tensorOptions(derivedDType, layout, device, requires_grad),
        memoryFormat.toNativeOptional
      )
    )

// TODO fullLike
// TODO quantize_per_tensor
// TODO quantize_per_channel
// TODO dequantize
// TODO complex
// TODO polar
// TODO heavside

  def pickleLoad(data: ByteVector): SeqMap[String, Tensor[DType]] =
    val dict: GenericDict = torchNative.pickle_load(data).toGenericDict()
    // We need to extract the members in one go or we risk too early deallocation of native objects here
    val buffer = new Array[(IValue, IValue)](dict.size().toInt)
    val nativeIt = dict.begin()
    for (i <- 0 until buffer.size)
      buffer(i) = (nativeIt.access().key(), nativeIt.access().value())
      nativeIt.increment()
    VectorMap.from(buffer.map { (key, value) =>
      // TODO better error handling
      (key.toStringRef().getString(), fromNative[DType](value.toTensor().clone()))
    })

  def pickle_load(path: Path) = pickleLoad(path)

  def pickleLoad(path: Path): Map[String, Tensor[DType]] =
    val data: ByteVector = Files.readAllBytes(path).asInstanceOf[ByteVector]
    pickleLoad(data)

  def save(model: ModelModule, filePath: String): Unit = {
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to(filePath) // "net.pt"
  }

  // torch.save(model.state_dict(), 'resnet.ckpt')
  def save(state_dict: Map[String, Tensor[?]], checkpoint_filePath: String): Unit = {
//    val archive = new OutputArchive
//    archive.save_to(filePath) //"net.pt"
    val byteArray = SerializeUtils.serialize(state_dict)
    Files.write(Paths.get(checkpoint_filePath), byteArray)
  }

  def pickle_save(tensors: SeqMap[String, Tensor[DType]]) = pickleSave(tensors)

  def pickleSave(tensors: SeqMap[String, Tensor[DType]]) =
    tensors.map { (k, v) =>
      (IValue(k), IValue(v.native))
    }

}

object SerializeUtils {

  import java.io.{ByteArrayOutputStream, ObjectOutputStream}

  def serialize(obj: Any): Array[Byte] = {
    val stream = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(stream)
    oos.writeObject(obj)
    oos.close()
    stream.toByteArray
  }
}
