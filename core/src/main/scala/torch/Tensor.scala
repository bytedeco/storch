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

import torch.numpy.matrix.NDArray
import org.bytedeco.javacpp.{
  Pointer,
  BoolPointer,
  BytePointer,
  DoublePointer,
  FloatPointer,
  IntPointer,
  LongPointer,
  ShortPointer
}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Tensor as NativeTensor,
  BoolOptional,
  TensorOptions,
  TensorBase,
  DoubleOptional,
  EllipsisIndexType,
  Generator,
  GeneratorOptional,
  LongOptional,
  NamedTensorMeta,
  Node,
  Quantizer,
  ScalarOptional,
  ScalarTypeOptional,
  Storage,
  SymInt,
  SymIntOptional,
  TensorArrayRefOptional,
  TensorIndex,
  TensorIndexArrayRef,
  TensorIndexVector,
  TensorOptional,
  TensorOptionalList,
  TensorTensorHook,
  TensorVector,
  VoidTensorHook
}
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch.ScalarType

import java.nio.{Buffer, ByteBuffer, DoubleBuffer, FloatBuffer, IntBuffer, LongBuffer, ShortBuffer}
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import torch.internal.NativeConverters.{toOptional, toScalar}
import spire.math.{Complex, UByte}
import torch.internal.NativeConverters
import torch.internal.NativeConverters.toArray
import Device.CPU
import torch.Layout.{Sparse, SparseBsc, SparseBsr, SparseCsc, SparseCsr, Strided}
import torch.internal.NativeConverters.{fromNative, toNative}

import scala.annotation.implicitNotFound
import torch.nn.functional as F

import scala.collection.mutable.ListBuffer

case class TensorTuple[D <: DType](
    values: Tensor[D],
    indices: Tensor[Int64]
)

trait TensorCreator[A, T <: DType] {
  def create(data: Seq[A]): Tensor[T]
}

object TensorCreator:
  given longCreator: TensorCreator[Long, Int64] with
    def create(data: Seq[Long]): Tensor[Int64] = torch.Tensor(data)

  given intCreator: TensorCreator[Int, Int32] with
    def create(data: Seq[Int]): Tensor[Int32] = torch.Tensor(data)

  given doubleCreator: TensorCreator[Double, Float64] with
    def create(data: Seq[Double]): Tensor[Float64] = torch.Tensor(data)

  given floatCreator: TensorCreator[Float, Float32] with
    def create(data: Seq[Float]): Tensor[Float32] = torch.Tensor(data)

/** A [[torch.Tensor]] is a multi-dimensional matrix containing elements of a single data type. */
sealed abstract class Tensor[D <: DType]( /* private[torch]  */ val native: pytorch.Tensor) {
  require(
    native.numel <= Int.MaxValue,
    s"Storch only supports tensors with up to ${Int.MaxValue} elements"
  )

  def new_tensor[A, T <: DType](data: Seq[A])(using creator: TensorCreator[A, T]): Tensor[T] = {
    creator.create(data)
  }

  def ==(other: ScalaType): Tensor[Bool] = eq(other)

  def add[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.add(toScalar(s))
  )

  def retain_grad = native.retain_grad()

  def retains_grad: Boolean = native.retains_grad

  def grad_fn: Node = grad_fn()

  def grad_fn(un_used: Int*): Node = {
    val gradFn = native.grad_fn()
    if gradFn != null then println(s"grad fn is  ${gradFn.getptr.name().getString}")
    gradFn
  }

  def gradFn(): Node = {
    val gradFn = native.grad_fn()
    if gradFn != null then println(s"gradient function is  ${gradFn.getptr.name().getString}")
    gradFn
  }

  def grad_fn_name: String = if native.grad_fn() != null then
    native.grad_fn().getptr.name().getString
  else
    "grad_fn is none ,can not invoke, please set required_grad with true or wait backward method excuted "

  def set_requires_grad(requires_grad: Boolean) = native.set_requires_grad(requires_grad)

  def setRequiresGrad(requires_grad: Boolean) = native.set_requires_grad(requires_grad)

  def requires_grad: Boolean = native.requires_grad()

  def requires_grad(unused: Int*): Boolean = native.requires_grad()

  def requires_grad_() = native.requires_grad_()

  def requires_grad_(required_grad: Boolean) = native.requires_grad_(required_grad)

  def is_leaf(un_used: Int*): Boolean = native.is_leaf()

  def output_nr(un_used: Int*): Long = native.output_nr()

  def retains_grad(un_used: Int*): Boolean = native.retains_grad()

  def _version(un_used: Int*): Long = native._version()

  def is_view(un_used: Int*): Boolean = native.is_view()

  def name(un_used: Int*): String = native.name().toString()

  def is_leaf: Boolean = native.is_leaf()

  def output_nr: Long = native.output_nr()

  def tensorBase = native._base()

  def _version: Long = native._version()

  def is_view: Boolean = native.is_view()

  def name: String = native.name().toString()

  def weak_use_count = native.weak_use_count()

  def use_count = native.use_count()

  def remove_hook(pos: Int): Unit = native.remove_hook(pos)

  def sizes: Seq[Int] = sizes()

  def sizes(un_used: Int*): Seq[Int] = {
    val longArrayRef = native.sizes()
    val sizesBuffer = new ListBuffer[Int]()
    for (i <- 0 until longArrayRef.size().toInt) {
      val element = longArrayRef.get(i.toLong)
      sizesBuffer.append(element.toInt)
    }
    sizesBuffer.toSeq

  }

  def strides: Seq[Int] = strides()

  def strides(un_used: Int*): Seq[Int] = {
    val stridesRef = native.strides()
    val stridesBuffer = new ListBuffer[Int]()
    for (i <- 0 until stridesRef.size().toInt) {
      val element = stridesRef.get(i.toLong)
      stridesBuffer.append(element.toInt)
    }
    stridesBuffer.toSeq
  }

  def set_data(data: Tensor[D]) = native.set_data(data.native)

  def data(un_used: Int*): Tensor[D] = fromNative(native.data())

  def ndimension(un_used: Int*): Long = native.ndimension()

  def nbytes(un_used: Int*): Long = native.nbytes()

  def itemsize(un_used: Int*): Long = native.itemsize()

  def element_size(un_used: Int*): Long = native.element_size()

  def has_storage(un_used: Int*): Boolean = native.has_storage()

  def data = fromNative(native.data())

  def share_memory = native.share_memory_()

  def ndimension: Long = native.ndimension()

  def nbytes: Long = native.nbytes()

  def itemsize: Long = native.itemsize()

  def element_size: Long = native.element_size()

  def has_storage: Boolean = native.has_storage()

  def has_names: Boolean = native.has_names()

  def is_alias_of(base: TensorBase): Boolean = native.is_alias_of(base)

  def get_named_tensor_meta = native.get_named_tensor_meta()

  def storage(un_used: Int*) = native.storage()

  def variable_data(un_used: Int*) = fromNative(native.variable_data())

  def tensor_data(un_used: Int*) = fromNative(native.tensor_data())

  def storage = native.storage()

  def variable_data = fromNative(native.variable_data())

  def tensor_data = fromNative(native.tensor_data())

  def register_hook(hook: VoidTensorHook) = native.register_hook(hook)

  import scala.annotation.targetName

  @targetName("register_void_hook")
  def register_hook[D <: DType](voidFunc: Tensor[D] => Unit) = {

    val voidHook = new VoidTensorHook {

      override def call(grad: TensorBase): Unit = {

        voidFunc(fromNative(NativeTensor(grad)))
      }
    }
    native.register_hook(voidHook)
  }

  @targetName("register_tensor_hook")
  def register_hook[D <: DType](tensorFunc: Tensor[D] => Tensor[D]) = {

    val scaleGradHook = new TensorTensorHook {

      override def call(grad: TensorBase): TensorBase = {

        tensorFunc(fromNative(NativeTensor(grad))).native
      }
    }
    native.register_hook(scaleGradHook)
  }

  def register_hook(hook: TensorTensorHook) = native.register_hook(hook)

  def key_set() = native.key_set()

  def data_ptr: Pointer = native.data_ptr()

  def scalar_type = native.scalar_type()

  def scalar_type(un_used: Int*) = native.scalar_type()

  def stride(dim: Int) = native.stride(dim.toLong)

  def size(dim: Long) = native.size(dim)

  def defined = native.defined()

  def reset = native.reset()

  def data_element = this match {
    case i: Tensor[Int32]   => this.data_ptr_int
    case f: Tensor[Float32] => this.data_ptr_float
    case d: Tensor[Float64] => this.data_ptr_double
    case l: Tensor[Int64]   => this.data_ptr_long
    case b: Tensor[Bool]    => this.data_ptr_bool
    case byte: Tensor[Byte] => this.data_ptr_byte
    case s: Tensor[Short]   => this.data_ptr_short
    case _                  => throw new IllegalArgumentException("Unsupported scalar type")
  }

  def data_element(un_used: Int*) = this.scalar_type.value match {
    case 11 => this.data_ptr_bool()
    case 0  => this.data_ptr_byte()
    case 1  => this.data_ptr_char()
    case 2  => this.data_ptr_short()
    case 3  => this.data_ptr_int()
    case 4  => this.data_ptr_long()
    case 5  => this.data_ptr_float()
    case 6  => this.data_ptr_float()
    case 7  => this.data_ptr_double()
    case _  => throw new IllegalArgumentException("Unsupported scalar type")
  }

  def mutable_data_ptr: Pointer = native.mutable_data_ptr()

  def const_data_ptr: Pointer = native.const_data_ptr()

  def options: TensorOptions = native.options()

  def quantizer: Quantizer = native.quantizer()

  def data_ptr_bool(un_used: Int*): Boolean = native.data_ptr_bool().get()

  def data_ptr_char(un_used: Int*): Byte = native.data_ptr_char().get()

  def data_ptr_byte(un_used: Int*): Byte = native.data_ptr_byte().get()

  def data_ptr_double(un_used: Int*): Double = native.data_ptr_double().get()

  def data_ptr_float(un_used: Int*): Float = native.data_ptr_float().get()

  def data_ptr_int(un_used: Int*): Int = native.data_ptr_int().get()

  def data_ptr_long(un_used: Int*): Long = native.data_ptr_long().get()

  def data_ptr_short(un_used: Int*): Short = native.data_ptr_short().get()

  def data_ptr_bool: Boolean = native.data_ptr_bool().get()

  def data_ptr_char: Byte = native.data_ptr_char().get()

  def data_ptr_byte: Byte = native.data_ptr_byte().get()

  def data_ptr_double: Double = native.data_ptr_double().get()

  def data_ptr_float: Float = native.data_ptr_float().get()

  def data_ptr_int: Int = native.data_ptr_int().get()

  def data_ptr_long: Long = native.data_ptr_long().get()

  def data_ptr_short: Short = native.data_ptr_short().get()

  def +[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = add(s)

  def add[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.add(other.native)
  )

  def +[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = add(other)

  // TODO add typelevel casting rules. I.e. An integral output tensor cannot accept a floating point tensor.
  // https://github.com/pytorch/pytorch/blob/041edeeecb75f3c110605d7311fa46abe1c62ea9/c10/core/ScalarType.h
  // https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
  def +=[D2 <: DType](other: Tensor[D2]): this.type =
    native.add_(other.native)
    this

  def +=[S <: ScalaType](s: S): this.type =
    native.add_(toScalar(s))
    this

  def add_[S <: ScalaType](s: S): this.type =
    native.add_(toScalar(s))
    this

  def sub[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.sub(toScalar(s))
  )

  def -[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = sub(s)

  def sub[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.sub(other.native)
  )

  def -[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = sub(other)

  def -=[D2 <: DType](other: Tensor[D2]): this.type =
    native.sub_(other.native)
    this

//  def sub_[D2 <: DType](other: Tensor[D2]): this.type =
//    native.sub_(other.native)
//    this
//
//  def sub_[S <: ScalaType](s: S): this.type =
//    native.sub_(toScalar(s))
//    this

  def -=[S <: ScalaType](s: S): this.type =
    native.sub_(toScalar(s))
    this

  def mul[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.mul(toScalar(s))
  )
//todo make sure the type of s is correct
//  def multiply[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
//    native.multiply(toScalar(s))
//  )
  // todo make sure the type of s is correct
  def multiply[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.multiply(other.native)
  )

  // todo make sure the type of s is correct
  def dot[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.dot(other.native)
  )

  // todo make sure the type of s is correct
  def vdot[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.vdot(other.native)
  )

  def new_empty[D <: DType](size: Seq[Int]): Tensor[D] = fromNative(
    native.new_empty(size.map(_.toLong)*)
  )

  def *[S <: ScalaType](s: S): Tensor[Promoted[D, ScalaToDType[S]]] = mul(s)

  def mul[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.mul(other.native)
  )

  def *[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = mul(other)

  def **[D2 <: DType](exponent: Tensor[D2])(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, D2] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, D2] NotEqual Complex32
  ): Tensor[Promoted[D, D2]] = fromNative(
    native.pow(exponent.native)
  )

  /** @see [[torch.pow]] */
  def **[S <: ScalaType](exponent: S)(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, ScalaToDType[S]] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, ScalaToDType[S]] NotEqual Complex32
  ): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.pow(exponent.toScalar)
  )

  def **=[S <: ScalaType](exponent: S): this.type = {
    native.pow_(toScalar(exponent))
    this
  }

  def **=[D1 <: DType](exponent: Tensor[D1]): this.type = {
    native.pow_(exponent.native)
    this
  }

  def *=[D2 <: DType](other: Tensor[D2]): this.type =
    native.mul_(other.native)
    this

  def *=[S <: ScalaType](s: S): this.type =
    native.mul_(toScalar(s))
    this

  def div[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.div(toScalar(s))
  )

  def /[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = div(s)

  def !=&[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.not_equal(toScalar(s))
  )

  def !=&[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = fromNative(
    native.not_equal(other.native)
  )

  def !==&[S <: ScalaType](s: S): this.type =
    native.not_equal_(toScalar(s))
    this

  def !==&[D2 <: DType](other: Tensor[D2]): this.type =
    native.not_equal_(other.native)
    this

  def !=[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.ne(toScalar(s))
  )

  def !=[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = fromNative(native.ne(other.native))

  def !==[S <: ScalaType](s: S): this.type =
    native.ne_(toScalar(s))
    this

  def !==[D2 <: DType](other: Tensor[D2]): this.type =
    native.ne_(other.native)
    this
  def !==^[S <: ScalaType](s: S)(using D <:< FloatNN): this.type =
    native.ne_(toScalar(s))
    this
  def !==^[D2 <: DType](other: Tensor[D2])(using D <:< FloatNN): this.type =
    native.ne_(other.native)
    this

  /** Divides each element of this tensor by the corresponding element of `other`. * */
  def div[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = fromNative(native.div(other.native))

  /** Divides each element of this tensor by the corresponding element of `other`. * */
  def /[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = div(other)

  def /=[D2 <: DType](other: Tensor[D2])(using D <:< FloatNN): this.type =
    native.div_(other.native)
    this

  def /=[S <: ScalaType](s: S)(using D <:< FloatNN): this.type =
    native.div_(toScalar(s))
    this

  def apply[T <: Boolean | Long: ClassTag](
      indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int32] | Tensor[Int64] |
        Seq[T] | None.type | Ellipsis)*
  ): Tensor[D] = index(indices*)

//  def apply[T <: Boolean | Long : ClassTag](
//                                             indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int64] | Seq[T] |
//                                               None.type | Ellipsis)*
//                                           ): Tensor[D] = index(indices *)

  /** Computes the absolute value of each element. */
  def abs: Tensor[D] = fromNative(native.abs())

  def acos: Tensor[D] = fromNative(native.acos())

  def adjoint: Tensor[D] = fromNative(native.adjoint())

  /** Tests if all elements of this tensor evaluate to `true`. */
  def all: Tensor[Bool] = fromNative(native.all())

  /** @see [[torch.allclose]] */
  def allclose(
      other: Tensor[?],
      rtol: Double = 1e-05,
      atol: Double = 1e-08,
      equalNan: Boolean = false
  ) = native.allclose(other.native, rtol, atol, equalNan)

  def any: Tensor[Bool] = fromNative(native.any())

  /** Tests if any element of this tensor evaluates to `true`. */
  def any(dim: Int, keepdim: Boolean = true): Tensor[Bool] = fromNative(native.any(dim, keepdim))

  /** Returns the indices of the maximum value of all elements in the tensor.
    *
    * This is the second value returned by torch.max(). See its documentation for the exact
    * semantics of this method.
    *
    * Example:
    * ```scala sc
    * val a = torch.rand(Seq(1, 3))
    * a.argmax()
    * // tensor dtype=float32, shape=[1] 2
    * ```
    *
    * @param dim
    * @param keepdim
    * @return
    */
  def argmax(dim: Int | Option[Int] = None, keepdim: Boolean = false): Tensor[Int64] = fromNative(
    native.argmax(NativeConverters.toOptional(dim), keepdim)
  )

  /** Computes the gradient of current tensor w.r.t. graph leaves.
    *
    * The graph is differentiated using the chain rule. If the tensor is non-scalar (i.e. its data
    * has more than one element) and requires gradient, the function additionally requires
    * specifying `gradient`. It should be a tensor of matching type and location, that contains the
    * gradient of the differentiated function w.r.t. `self`.
    *
    * This function accumulates gradients in the leaves - you might need to zero `.grad` attributes
    * or set them to `None` before calling it. See `Default gradient layouts<default-grad-layouts>`
    * for details on the memory layout of accumulated gradients.
    *
    * Note
    *
    * If you run any forward ops, create `gradient`, and/or call `backward` in a user-specified CUDA
    * stream context, see `Stream semantics of backward passes<bwd-cuda-stream-semantics>`.
    *
    * Note
    *
    * When `inputs` are provided and a given input is not a leaf, the current implementation will
    * call its grad_fn (though it is not strictly needed to get this gradients). It is an
    * implementation detail on which the user should not rely. See
    * <https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780> for more details.
    */
  def backward: Unit = {
    if !this.requiresGrad then this.requiresGrad_=(true)
    native.backward()
  }

  def backward(un_used: Int*): Unit = {
    if !this.requiresGrad then this.requiresGrad_=(true)
    native.backward()
  }

//  def backward(retain_graph: Boolean = true, create_graph: Boolean = false): Unit = {
//    if !this.requiresGrad then this.requiresGrad_=(true)
//    native.backward()
//  }
  def backward(
      gradient: Tensor[D] | Option[Tensor[D]] = None,
      retain_graph: Option[Boolean] | Boolean = Some(true),
      create_graph: Boolean = false,
      inputs: Option[Seq[Tensor[D]]] = None
  ): Unit = {
    if !this.requiresGrad then this.requiresGrad_=(true)
//    val nativeRetainGraph =
//      if retain_graph.isDefined then new BoolOptional(retain_graph.get) else new BoolOptional()
//
    val nativeGradient = gradient match {
      case t: Tensor[D] => t.native
      case Some(t)      => t.native
      case None         => torch.empty(0).native
    }
    val nativeRetainGraph = retain_graph match {
      case Some(b)    => new BoolOptional(b)
      case None       => new BoolOptional()
      case b: Boolean => new BoolOptional(b)
    }
    val nativeInputsRef = if inputs.isDefined then
      val tensorVector = new TensorVector(inputs.get.map(_.native)*)
      new TensorArrayRefOptional(tensorVector)
    else new TensorArrayRefOptional()
    native.backward(nativeGradient, nativeRetainGraph, create_graph, nativeInputsRef)
  }

  /** Returns a new Tensor, detached from the current graph.
    *
    * The result will never require gradient.
    *
    * This method also affects forward mode AD gradients and the result will never have forward mode
    * AD gradients.
    */
  def detach: Tensor[D] = fromNative(native.detach())

  def detach(un_used: Int*): Tensor[D] = fromNative(native.detach())

  /** Returns a copy of `input`.
    *
    * @note
    *   This function is differentiable, so gradients will flow back from the result of this
    *   operation to `input`. To create a tensor without an autograd relationship to `input` see
    *   `Tensor.detach`.
    */
  def clone(memoryFormat: MemoryFormat = MemoryFormat.Preserve): Tensor[D] =
    fromNative(native.clone(memoryFormat.toNativeOptional))

  def contiguous: Tensor[D] = fromNative(native.contiguous())

  def contiguous(un_used: Int*): Tensor[D] = fromNative(native.contiguous())

  /** Copies the elements from `src` into this tensor and returns this.
    *
    * The `src` tensor must be broadcastable with the self tensor. It may be of a different data
    * type or reside on a different device.
    *
    * @param src
    *   the source tensor to copy from
    * @param nonBlocking
    *   if `true` and this copy is between CPU and GPU, the copy may occur asynchronously with
    *   respect to the host. For other cases, this argument has no effect.
    */
  def copy_(src: Tensor[?], nonBlocking: Boolean = false): this.type =
    native.copy_(src.native, nonBlocking)
    this

  /** Returns a new tensor with the sine of the elements of this tensor. */
  def cos: Tensor[FloatPromoted[D]] = fromNative(native.cos())

  def device: Device = Device(native.device())

  def dim: Int = native.dim().toInt

  def dtype: D

  /** Computes element-wise equality */
  def eq(other: ScalaType): Tensor[Bool] = fromNative(native.eq(toScalar(other)))

  /** Computes element-wise equality
    *
    * The argument can be a tensor whose shape is broadcastable with this tensor.
    */
  def eq(other: Tensor[?]): Tensor[Bool] = fromNative(native.eq(other.native))

  def ==(other: Tensor[?]): Tensor[Bool] = eq(other)

  def ==#[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.eq(toScalar(s))
  )

  def ==#[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = fromNative(native.eq(other.native))

  def ===[S <: ScalaType](s: S): this.type =
    native.eq_(toScalar(s))
    this

  def ===[D2 <: DType](other: Tensor[D2]): this.type =
    native.eq_(other.native)
    this

  def ===&[S <: ScalaType](s: S)(using D <:< FloatNN): this.type =
    native.eq_(toScalar(s))
    this

  def ===&[D2 <: DType](other: Tensor[D2])(using D <:< FloatNN): this.type =
    native.eq_(other.native)
    this

  override def equals(that: Any): Boolean =
    that match
      case other: Tensor[?] if dtype == other.dtype => native.equal(other.native)
      case _                                        => false

  /** True if `other` has the same size and elements as this tensor, false otherwise. */
  def equal(other: Tensor[D]): Boolean = native.equal(other.native)

  /** Returns the tensor with elements exponentiated. */
  def exp(un_used: Int*): Tensor[D] = fromNative(native.exp())

//  def ^e: Tensor[D] = fromNative(native.exp())
  /** Returns a new view of this tensor with singleton dimensions expanded to a larger size.
    *
    * Passing -1 as the size for a dimension means not changing the size of that dimension.
    *
    * Tensor can be also expanded to a larger number of dimensions, and the new ones will be
    * appended at the front. For the new dimensions, the size cannot be set to -1.
    *
    * Expanding a tensor does not allocate new memory, but only creates a new view on the existing
    * tensor where a dimension of size one is expanded to a larger size by setting the `stride` to
    * 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new
    * memory.
    *
    * @param sizes
    *   the desired expanded size
    *
    * @note
    *   More than one element of an expanded tensor may refer to a single memory location. As a
    *   result, in-place operations (especially ones that are vectorized) may result in incorrect
    *   behavior. If you need to write to the tensors, please clone them first.
    *
    * @example
    *   ```scala sc
    *   val x = torch.tensor((Seq(Seq(1), Seq(2), Seq(3)))
    *   x.size // [3, 1]
    *   x.expand(3, 4)
    *   x.expand(-1, 4) // -1 means not changing the size of that dimension
    *   ```
    */
  def expand(sizes: Int*) = fromNative(native.expand(sizes.map(_.toLong)*))

  def corrcoef(un_used: Int*): Tensor[D] = fromNative(native.corrcoef())

  def flatten: Tensor[D] = fromNative(native.flatten())

  def flatten(un_used: Int*): Tensor[D] = fromNative(native.flatten())

  def flatten(startDim: Int = 0, endDim: Int = -1): Tensor[D] = fromNative(
    native.flatten(startDim, endDim)
  )

  def float: Tensor[Float32] = to(dtype = float32)

  def bools: Tensor[Bool] = to(dtype = bool)

  def boolean: Tensor[Bool] = to(dtype = bool)

  def double: Tensor[Float64] = to(dtype = float64)

  def float(un_used: Int*): Tensor[Float32] = to(dtype = float32)

  def bools(un_used: Int*): Tensor[Bool] = to(dtype = bool)

  def boolean(un_used: Int*): Tensor[Bool] = to(dtype = bool)

  def double(un_used: Int*): Tensor[Float64] = to(dtype = float64)

  def long(un_used: Int*): Tensor[Int64] = to(dtype = int64)

  def int(un_used: Int*): Tensor[Int32] = to(dtype = int32)

  def int: Tensor[Int32] = to(dtype = int32)

  /** Divides each element of this tensor by `s` and floors the result. */
  def floorDivide[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.floor_divide(toScalar(s))
  )

  def floor_divide[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.floor_divide(toScalar(s))
  )

  /** Divides each element of this tensor by the corresponding element of `other` and floors the
    * result.
    */
  def floorDivide[D2 <: DType](other: Tensor[D2]): Tensor[Div[D, D2]] = fromNative(
    native.floor_divide(other.native)
  )

  /** This function returns an undefined tensor by default and returns a defined tensor the first
    * time a call to backward() computes gradients for this Tensor. The attribute will then contain
    * the gradients computed and future calls to backward() will accumulate (add) gradients into it.
    */
  def grad: Option[Tensor[D]] =
    val nativeGrad = native.grad()
    Option.when(nativeGrad.defined())(fromNative(nativeGrad))

  def grad(un_used: Int*): Option[Tensor[D]] =
    val nativeGrad = native.grad()
    Option.when(nativeGrad.defined())(fromNative(nativeGrad))

  def ge(other: ScalaType): Tensor[Bool] = fromNative(native.ge(toScalar(other)))

  def >=(other: ScalaType): Tensor[Bool] = ge(other)

//  def ge[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.ge(other.native))

  def >=[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.ge(other.native))

  def gt(other: ScalaType): Tensor[Bool] = fromNative(native.gt(toScalar(other)))

  def >(other: ScalaType): Tensor[Bool] = gt(other)

  def >>(other: ScalaType): Boolean = {
    val boolTensor = gt(other)
    boolTensor.all().item()
  }

  def <<(other: ScalaType): Boolean = {
    val boolTensor = lt(other)
    boolTensor.all().item()
  }

//  def gt[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.gt(other.native))

  def >[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.gt(other.native))

  def sym_nbytes = native.sym_nbytes()

  def sym_numel = native.sym_numel()

  def sym_storage_offset = native.sym_storage_offset()

  def suggest_memory_format = native.suggest_memory_format()

  def isContiguous: Boolean = native.is_contiguous()

  def is_contiguous = native.is_contiguous()

  def is_contiguous(un_used: Int*): Boolean = native.is_contiguous()

  def is_contiguous(memoryFormat: MemoryFormat) = native.is_contiguous(memoryFormat.toNative)

  def cuda(un_used: Int*): Tensor[D] = fromNative(native.cuda())

  def cpu(un_used: Int*): Tensor[D] = fromNative(native.cpu())

  def hip(un_used: Int*): Tensor[D] = fromNative(native.hip())

  def ve(un_used: Int*): Tensor[D] = fromNative(native.ve())

  def vulkan(un_used: Int*): Tensor[D] = fromNative(native.vulkan())

  def metal(un_used: Int*): Tensor[D] = fromNative(native.metal())

  def meta(un_used: Int*): Tensor[D] = fromNative(native.meta())

  def cuda: Tensor[D] = fromNative(native.cuda())

  def cpu: Tensor[D] = fromNative(native.cpu())

  def hip: Tensor[D] = fromNative(native.hip())

  def ve: Tensor[D] = fromNative(native.ve())

  def vulkan: Tensor[D] = fromNative(native.vulkan())

  def metal: Tensor[D] = fromNative(native.metal())

  def meta: Tensor[D] = fromNative(native.meta())

  def isCuda: Boolean = native.is_cuda()

  def is_neg: Boolean = native.is_neg()

  def is_cuda: Boolean = native.is_cuda()

  def is_cpu: Boolean = native.is_cpu()

  def is_xpu: Boolean = native.is_xpu()

  def is_ipu: Boolean = native.is_ipu()

  def is_hpu: Boolean = native.is_hpu()

  def is_xla: Boolean = native.is_xla()

  def is_privateuseone: Boolean = native.is_privateuseone()

  def is_hip: Boolean = native.is_hip()

  def is_ve: Boolean = native.is_ve()

  def is_mtia: Boolean = native.is_mtia()

  def is_lazy: Boolean = native.is_lazy()

  def is_mkldnn: Boolean = native.is_mkldnn()

  def is_vulkan: Boolean = native.is_vulkan()

  def is_metal: Boolean = native.is_metal()

  def is_maia: Boolean = native.is_maia()

  def is_meta: Boolean = native.is_meta()

  def is_inference: Boolean = native.is_inference()

  def is_nested: Boolean = native.is_nested()

  def is_mps: Boolean = native.is_mps()

  def is_sparse_csr: Boolean = native.is_sparse_csr()

  def is_sparse_native: Boolean = native.is_sparse()

  def isQuantized: Boolean = native.is_quantized()

  def isnan: Tensor[Bool] = fromNative(native.isnan())

  def isNonzero: Boolean = native.is_nonzero()

  def isConj: Boolean = native.is_conj()

  def isSparse: Boolean = native.is_sparse()

  def clip(un_used: Int*): Tensor[D] = fromNative(native.clip())

  def clamp(un_used: Int*): Tensor[D] = fromNative(native.clamp())

  def clip: Tensor[D] = fromNative(native.clip())

  def clamp: Tensor[D] = fromNative(native.clamp())

  def item(un_used: Int*): DTypeToScala[D] = item

  // TODO override in subclasses instead?
  def item: DTypeToScala[D] =
    import ScalarType.*
    val out = native.dtype().toScalarType().intern() match
      case Byte        => UByte(native.item_int())
      case Char        => native.item_char()
      case Short       => native.item_short()
      case Int         => native.item_int()
      case Long        => native.item_long()
      case Half        => native.item().toHalf.asFloat()
      case Float       => native.item_float()
      case Double      => native.item_double()
      case ComplexHalf => ??? // TODO how to access complex scalar values?
      case ComplexFloat =>
        val b = native.contiguous.createBuffer[FloatBuffer]
        Complex(b.get(), b.get())
      case ComplexDouble =>
        val b = native.contiguous.createBuffer[DoubleBuffer]
        Complex(b.get(), b.get())
      case Bool                   => native.item().toBool
      case QInt8                  => native.item_char()
      case QUInt8                 => native.item_short()
      case QInt32                 => native.item_int()
      case BFloat16               => native.item().toBFloat16.asFloat()
      case QUInt4x2               => ???
      case QUInt2x4               => ???
      case Bits1x8                => ???
      case Bits2x4                => ???
      case Bits4x2                => ???
      case Bits8                  => ???
      case Bits16                 => ???
      case Float8_e5m2            => native.item().toFloat8_e5m2().asFloat()
      case Float8_e4m3fn          => native.item().toFloat8_e4m3fn().asFloat()
      case Undefined | NumOptions => ???
    out.asInstanceOf[DTypeToScala[D]]

  def layout: Layout = Layout.fromNative(native.layout())

  /** Returns the tensor with elements logged. */
  def log: Tensor[D] = fromNative(native.log())

  def long: Tensor[Int64] = to(dtype = int64)

  def le(other: ScalaType): Tensor[Bool] = fromNative(native.le(toScalar(other)))

  def <=(other: ScalaType): Tensor[Bool] = le(other)

  def <=[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.le(other.native))

  def lt(other: ScalaType): Tensor[Bool] = fromNative(native.lt(toScalar(other)))

  def <(other: ScalaType): Tensor[Bool] = lt(other)

  def <[D <: DType](other: Tensor[D]): Tensor[Bool] = fromNative(native.lt(other.native))

  def clip[S <: ScalaType](min: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.clip(new ScalarOptional(toScalar(min)))
  )

  def clip[S <: ScalaType](min: S, max: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.clip(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
  )

  def clamp[S <: ScalaType](min: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.clamp(new ScalarOptional(toScalar(min)))
  )

  def clamp[S <: ScalaType](min: S, max: S): Tensor[Div[D, ScalaToDType[S]]] = fromNative(
    native.clamp(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
  )

  def matmul[D2 <: DType](u: Tensor[D2]): Tensor[Promoted[D, D2]] =
    fromNative(native.matmul(u.native))

  def `@`[D2 <: DType](u: Tensor[D2]): Tensor[Promoted[D, D2]] = matmul(u)

  def @@[D2 <: DType](u: Tensor[D2]): Tensor[Promoted[D, D2]] = matmul(u)

  def norm: Tensor[D] = fromNative(native.norm())

  def norm(un_used: String*): Tensor[D] = fromNative(native.norm())

  def norm[S <: ScalaType](s: S): Tensor[Div[D, ScalaToDType[S]]] = {
    val sFloat = s match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(
      native.norm(toScalar(sFloat))
    )
  }

  def norm[D1 <: DType, S <: ScalaType](p: S, dim: Int*): Tensor[D1] = {

    val pFloat = p match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(native.norm(ScalarOptional(toScalar(pFloat)), dim.map(_.toLong)*))
  }

  def norm[D1 <: DType, S <: ScalaType](
      p: S,
      dim: Seq[Long],
      keepdim: Boolean = false
  ): Tensor[D1] = {
    val pFloat = p match {
      case m: Float  => m
      case m: Double => m.toFloat
      case m: Int    => m.toFloat
      case m: Long   => m.toFloat
    }
    fromNative(native.norm(ScalarOptional(toScalar(pFloat)), dim.toArray, keepdim))
  }

  /** Fills elements of self tensor with value where mask is `true`. The shape of mask must be
    * [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)
    * with the shape of the underlying tensor.
    *
    * @param mask
    *   the boolean mask
    * @param value
    *   the value to fill in with
    * @return
    *   Tensor with masked elements set to `value`
    */

  def masked_fill[S <: ScalaType](
      mask: Tensor[Bool],
      value: S
  ): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(native.masked_fill(mask.native, toScalar(value)))

  def maskedFill[S <: ScalaType](
      mask: Tensor[Bool],
      value: S
  ): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(native.masked_fill(mask.native, toScalar(value)))

  /** Returns the maximum value of all elements of this tensor. */
  def max(un_used: Int*): Tensor[D] = fromNative(native.max())

  def max: Tensor[D] = fromNative(native.max())

  /** Returns a tuple ``(values, indices)`` where ``values`` is the maximum value of each row of the
    * `input` tensor in the given dimension `dim`. And ``indices`` is the index location of each
    * maximum value found (argmax).
    *
    * If ``keepdim`` is ``true``, the output tensors are of the same size as ``input`` except in the
    * dimension ``dim`` where they are of size 1. Otherwise, ``dim`` is squeezed (see
    * :func:`torch.squeeze`), resulting in the output tensors having 1 fewer dimension than
    * ``input``.
    *
    * @note
    *   If there are multiple maximal values in a reduced row then the indices of the first maximal
    *   value are returned.
    */
  def max(dim: Int, keepdim: Boolean = false): TensorTuple[D] =
    val nativeTuple = native.max(dim.toLong, keepdim)
    TensorTuple(values = fromNative(nativeTuple.get0), indices = new Int64Tensor(nativeTuple.get1))

  def maximum[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    fromNative[Promoted[D, D2]](native.maximum(other.native))

  def mean: Tensor[D] = fromNative(native.mean())

  def mean(un_used: Int*): Tensor[D] = fromNative(native.mean())

  /** @see
    *   [[torch.mean]]
    */
  def mean[D2 <: DType | Derive](
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    val derivedDType = dtype match
      case _: Derive => this.dtype
      case d: DType  => d
    fromNative(
      torchNative.mean(
        native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

  def min: Tensor[Int64] = fromNative(native.min())

  def min(un_used: Int*): Tensor[Int64] = fromNative(native.min())

  def minimum[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] =
    fromNative[Promoted[D, D2]](native.minimum(other.native))

  /** Accessing this property is equivalent to calling adjoint(). */
  def mH: Tensor[D] = fromNative(native.mH())

  /** Returns a view of this tensor with the last two dimensions transposed.
    *
    * `x.mT` is equivalent to `x.transpose(-2, -1)`.
    */
  def mT: Tensor[D] = fromNative(native.mT())

  /** Returns a new tensor with the negative of the elements of this tensor. */
  def neg: Tensor[D] = fromNative(native.neg())

  /** Returns the total number of elements in the input tensor. */
  def numel: Long = native.numel()

  def permute(dims: Int*): Tensor[D] = fromNative(native.permute(dims.map(_.toLong)*))

  /** @see [[torch.pow]] */
  def pow[D2 <: DType](exponent: Tensor[D2])(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, D2] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, D2] NotEqual Complex32
  ): Tensor[Promoted[D, D2]] = fromNative(
    native.pow(exponent.native)
  )

  /** @see [[torch.pow]] */
  def pow[S <: ScalaType](exponent: S)(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, ScalaToDType[S]] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, ScalaToDType[S]] NotEqual Complex32
  ): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.pow(exponent.toScalar)
  )

  def prod[D <: DType](dtype: D = this.dtype) = fromNative(native.prod())

  /** Repeats this tensor along the specified dimensions.
    *
    * Unlike [[expand]], this function copies the tensor’s data.
    *
    * @param sizes
    *   The number of times to repeat this tensor along each dimension
    */
  def repeat(sizes: Int*): Tensor[D] = fromNative(native.repeat(sizes.map(_.toLong)*))

  def reshape(shape: Int*): Tensor[D] = fromNative(native.reshape(shape.map(_.toLong)*))

  def shape: Seq[Int] = size

  def softmax[Out <: FloatNN | Derive](
      dim: Int,
      dtype: Out = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, Out]] = F.softmax(input = this, dim = dim, dtype = dtype)

  def square = fromNative(native.square())

  def squeeze: Tensor[D] = fromNative(native.squeeze())

  def size: Seq[Int] = ArraySeq.unsafeWrapArray(native.sizes.vec.get.map(_.toInt))

  def std: Tensor[D] = fromNative(native.std())

  /** Returns a new tensor with the sine of the elements of this tensor. */
  def sin: Tensor[FloatPromoted[D]] = fromNative(native.sin())

  /** Returns the sum of all elements of this tensor. */
  def sum: Tensor[Sum[D]] = fromNative(native.sum())
  def sum[D2 <: DType | Derive](
      dim: Int | Seq[Int] = Seq.empty,
      keepdim: Boolean = false,
      dtype: D2 = derive
  ): Tensor[DTypeOrDeriveFromTensor[D, D2]] =
    val derivedDType = dtype match
      case _: Derive => this.dtype
      case d: DType  => d
    fromNative(
      torchNative.sum(
        native,
        dim.toArray,
        keepdim,
        new ScalarTypeOptional(derivedDType.toScalarType)
      )
    )

  /** Expects `input` to be \<= 2-D tensor and transposes dimensions 0 and 1.
    *
    * 0-D and 1-D tensors are returned as is. When input is a 2-D tensor this is equivalent to
    * `transpose(input, 0, 1)`.
    */
  def t: Tensor[D] = fromNative(native.t())

  def T: Tensor[D] = fromNative(native.t())

  def t(un_used: Int*): Tensor[D] = fromNative(native.t())

  /** Returns a tensor that is a transposed version of `input` (this Tensor). The given dimensions
    * `dim0` and `dim1` are swapped.
    *
    * If `input` is a strided tensor then the resulting `out` tensor shares its underlying storage
    * with the `input` tensor, so changing the content of one would change the content of the other.
    *
    * If `input` is a [[https://pytorch.org/docs/stable/sparse.html#sparse-docs sparse tensor]] then
    * the resulting `out` tensor does not share the underlying storage with the input tensor.
    *
    * If input is a [[https://pytorch.org/docs/stable/sparse.html#sparse-docs sparse tensor]] with
    * compressed layout (SparseCSR, SparseBSR, SparseCSC or SparseBSC) the arguments `dim0` and
    * `dim1` must be both batch dimensions, or must both be sparse dimensions. The batch dimensions
    * of a sparse tensor are the dimensions preceding the sparse dimensions.
    *
    * @note
    *   Transpositions which interchange the sparse dimensions of a *SparseCSR* or *SparseCSC*
    *   layout tensor will result in the layout changing between the two options. Transposition of
    *   the sparse dimensions of a `SparseBSR` or `SparseBSC` layout tensor will likewise generate a
    *   result with the opposite layout.
    *
    * @example:
    *   {{{
    *  val x = torch.randn(2, 3)
    *  println(x)
    *  val y = torch.transpose(x, 0, 1)
    *  println(y)
    *   }}}
    *
    * @param input
    *   the input tensor.
    * @param dim0
    *   the first dimension to be transposed
    * @param dim1
    *   the second dimension to be transposed
    * @return
    *   Tensor[D]
    *
    * @see
    *   [[Tensor.mT]]
    */
  def transpose(dim0: Int, dim1: Int): Tensor[D] = fromNative(native.transpose(dim0, dim1))

  /** Calculates the variance of all elements of this tensor. */
  def variance = fromNative(native.`var`())

  /** Returns a new tensor with the negative of the elements of this tensor. */
  def unary_- : Tensor[D] = fromNative(native.neg())

  /** Returns a new tensor with a dimension of size one inserted at the specified position.
    *
    * The returned tensor shares the same underlying data with this tensor.
    *
    * A `dim` value within the range `[-input.dim() - 1, input.dim() + 1)` can be used. Negative
    * `dim` will correspond to [[unsqueeze]] applied at `dim` = `dim + input.dim() + 1`.
    *
    * Example:
    *
    * ```scala sc
    * val x = torch.Tensor(Seq(1, 2, 3, 4))
    * x.unsqueeze(0)
    * // [[1, 2, 3, 4]]
    * x.unsqueeze(1)
    * // [[1],
    * //  [2],
    * //  [3],
    * //  [4]]
    * ```
    *
    * @param dim
    *   the index at which to insert the singleton dimension
    */
  def unsqueeze(dim: Int): Tensor[D] = fromNative(native.unsqueeze(dim))

  def zero_(): this.type =
    native.zero_()
    this

  private def nativeIndices[T <: Boolean | Long: ClassTag](
      indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int32] | Tensor[Int64] |
        Seq[T] | None.type | Ellipsis)*
  ): TensorIndexArrayRef =
    def toSymInt(maybeInt: Option[Int]) = maybeInt.map(l => SymIntOptional(SymInt(l))).orNull
    // see https://pytorch.org/cppdocs/notes/tensor_indexing.html
    val nativeIndices: Seq[pytorch.TensorIndex] =
      for (i <- indices) yield i match
        case None =>
          new pytorch.TensorIndex()
        case i: Tensor[?] =>
          new pytorch.TensorIndex(i.native)
        case singleton: Int =>
          new pytorch.TensorIndex(singleton)
        case singleton: Long =>
          new pytorch.TensorIndex(singleton)
        case Slice(start, end, step) =>
          new pytorch.TensorIndex(
            new pytorch.Slice(toSymInt(start), toSymInt(end), toSymInt(step))
          )
        case s: Seq[T] @unchecked => new pytorch.TensorIndex(Tensor[T](s).native)
        case e: Ellipsis          => new pytorch.TensorIndex(new EllipsisIndexType)
        // TODO index with single boolean. Needs investigation why it is not mapped.
    new pytorch.TensorIndexArrayRef(new pytorch.TensorIndexVector(nativeIndices.toArray*))

  def index[T <: Boolean | Long: ClassTag](
      indices: (Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int32] | Tensor[Int64] |
        Seq[T] | None.type | Ellipsis)*
  ): Tensor[D] =
    fromNative(native.index(nativeIndices(indices*)))

  /** Set tensor value(s) at indices
    *
    * @example
    *   ```scala sc
    *   val t = torch.zeros(Seq(2, 2))
    *   // set first row to ones
    *   t(Seq(0)) = 1
    *   ```
    */
  def update[T <: Boolean | Long: ClassTag](
      indices: Seq[
        Slice | Int | Long | Tensor[Bool] | Tensor[UInt8] | Tensor[Int64] | Seq[T] | None.type |
          Ellipsis
      ] | Tensor[Bool] | Tensor[Int64] | Tensor[Int32] | Tensor[UInt8],
      values: Tensor[?] | ScalaType
  ): this.type =
    values match
      case t: Tensor[?] =>
        indices match {
          case i: Tensor[Bool]          => native.index_put_(nativeIndices(Seq(i)*), t.native)
          case i: Tensor[Int64]         => native.index_put_(nativeIndices(Seq(i)*), t.native)
          case i: Tensor[Int32]         => native.index_put_(nativeIndices(Seq(i)*), t.native)
          case i: Tensor[UInt8]         => native.index_put_(nativeIndices(Seq(i)*), t.native)
          case s: Seq[Slice] @unchecked => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Int] @unchecked   => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Long] @unchecked  => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Tensor[Bool]] @unchecked  => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Tensor[Int64]] @unchecked => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Tensor[UInt8]] @unchecked => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Seq[T]] @unchecked        => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[None.type] @unchecked     => native.index_put_(nativeIndices(s*), t.native)
          case s: Seq[Ellipsis] @unchecked      => native.index_put_(nativeIndices(s*), t.native)
          case _ => throw new IllegalArgumentException("Invalid Tensor indices")
        }

      case s: ScalaType @unchecked =>
        indices match {
          case i: Tensor[Bool]            => native.index_put_(nativeIndices(Seq(i)*), s.toScalar)
          case i: Tensor[Int64]           => native.index_put_(nativeIndices(Seq(i)*), s.toScalar)
          case i: Tensor[Int32]           => native.index_put_(nativeIndices(Seq(i)*), s.toScalar)
          case i: Tensor[UInt8]           => native.index_put_(nativeIndices(Seq(i)*), s.toScalar)
          case seq: Seq[Slice] @unchecked => native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Int] @unchecked   => native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Long] @unchecked  => native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Tensor[Bool]] @unchecked =>
            native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Tensor[Int64]] @unchecked =>
            native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Tensor[UInt8]] @unchecked =>
            native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[None.type] @unchecked => native.index_put_(nativeIndices(seq*), s.toScalar)
          case seq: Seq[Ellipsis] @unchecked  => native.index_put_(nativeIndices(seq*), s.toScalar)
          case _ => throw new IllegalArgumentException("Invalid ScalarType indices")
        }
    this

  def requiresGrad: Boolean = native.requires_grad()

  def requiresGrad_=(requiresGrad: Boolean): this.type =
    native.requires_grad_(requiresGrad)
    this

  def requiresGrad_(requiresGrad: Boolean): this.type =
    native.requires_grad_(requiresGrad)
    this

  def requires_grad_=(requiresGrad: Boolean): this.type =
    native.requires_grad_(requiresGrad)
    this

  def split_no_dim(
      splitSize: Int | Seq[Int]
  ): Seq[Tensor[D]] = {
    val result =
      splitSize match {
        case i: Int      => native.split(i.toLong)
        case s: Seq[Int] => native.split(s.map(_.toLong).toArray*)
      }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def split(
      splitSize: Int | Seq[Int],
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result =
      splitSize match {
        case i: Int      => native.split(i.toLong, dim.toLong)
        case s: Seq[Int] => native.split(s.map(_.toLong).toArray, dim.toLong)
      }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def unsafe_split_no_dim(
      splitSize: Int
  ): Seq[Tensor[D]] = {
    val result =
      splitSize match {
        case i: Int => native.unsafe_split(i.toLong)
      }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def unsafe_split(
      splitSize: Int,
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result =
      splitSize match {
        case i: Int => native.unsafe_split(i.toLong, dim.toLong)
      }
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def chunk(
      chunks: Int,
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result = native.chunk(chunks.toLong, dim.toLong)
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def unsafe_chunk(
      chunks: Int,
      dim: Int = 0
  ): Seq[Tensor[D]] = {
    val result = native.unsafe_chunk(chunks.toLong, dim.toLong)
    (0L until result.size()).map(i => fromNative(result.get(i)))
  }

  def take(indices: Tensor[Int64] | Tensor[Int32]): Tensor[D] = {
    indices.dtype match
      case torch.int64 => fromNative(native.take(indices.native))
      case torch.int32 => fromNative(native.take(indices.to(dtype = torch.int64).native))
  }

  def take_along_dim(indices: Tensor[Int64] | Tensor[Int32], dim: Int) = {

    indices.dtype match
      case torch.int64 => native.take_along_dim(indices.native, toOptional(dim))
      case torch.int32 =>
        native.take_along_dim(indices.to(dtype = torch.int64).native, toOptional(dim))
  }

  def to(device: Device | Option[Device]): Tensor[D] = device match
    case dev: Device => to(dev, this.dtype)
    case Some(dev)   => to(dev, this.dtype)
    case None        => this

  def to[U <: DType](dtype: U): Tensor[U] = to(this.device, dtype)

  // TODO support memory_format
  /** Performs Tensor dtype and/or device conversion. */
  def to[U <: DType](
      device: Device = this.device,
      dtype: U = this.dtype,
      nonBlocking: Boolean = false,
      copy: Boolean = false
  ): Tensor[U] =
    val targetDType = dtype.toScalarType
    if dtype == this.dtype && device == this.device && !copy then this.asInstanceOf[Tensor[U]]
    else if device == this.device then
      fromNative(
        native.to(
          targetDType,
          nonBlocking,
          copy,
          pytorch.MemoryFormatOptional(torchNative.MemoryFormat.Preserve)
        )
      )
    else
      fromNative(
        native.to(
          device.toNative,
          targetDType,
          nonBlocking,
          copy,
          pytorch.MemoryFormatOptional(torchNative.MemoryFormat.Preserve)
        )
      )

  def numpy[D <: DType]()(implicit
      ct: ClassTag[DTypeToScala[D]] = this.scalarClassTag
  ): NDArray[DTypeToScala[D]] = {
    val cpuTensor = to(device = CPU)
    // get data array and shape info
    val dataArray: Array[DTypeToScala[D]] = cpuTensor.toArray.asInstanceOf[Array[DTypeToScala[D]]]
    val shape = cpuTensor.size
    val ndim = shape.length
    NDArray(dataArray, shape, ndim, this.toNumpyDType).reshape(shape*)

  }

  def scalarClassTag: ClassTag[DTypeToScala[D]] = this.dtype match {
    case Float                          => ClassTag.Float.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Double                         => ClassTag.Double.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Byte                           => ClassTag.Byte.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Short                          => ClassTag.Short.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Int                            => ClassTag.Int.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Long                           => ClassTag.Long.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case Boolean                        => ClassTag.Boolean.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.float16 | DType.bfloat16 => ClassTag.Float.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.float32                  => ClassTag.Float.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.float64                  => ClassTag.Double.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.int8 | DType.uint8       => ClassTag.Byte.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.int16                    => ClassTag.Short.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.int32                    => ClassTag.Int.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.int64                    => ClassTag.Long.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.bool                     => ClassTag.Boolean.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.complex64                => ClassTag.Double.asInstanceOf[ClassTag[DTypeToScala[D]]]
    case DType.complex128               => ClassTag.Double.asInstanceOf[ClassTag[DTypeToScala[D]]]
//    case Complex[Float] => ClassTag(classOf[Complex[Float]]).asInstanceOf[ClassTag[DTypeToScala[D]]]
//    case _: Complex128.type => ClassTag(classOf[Complex[Double]]).asInstanceOf[ClassTag[DTypeToScala[D]]]
    case _ =>
      throw new UnsupportedOperationException(
        s"Unsupported dtype for numpy conversion: ${this.dtype}"
      )
  }

  def toNDArray[D <: DType]()(implicit ct: ClassTag[DTypeToScala[D]]): NDArray[DTypeToScala[D]] = {
    // move tensor to cpu
    val cpuTensor = to(device = CPU)
    // get data array and shape info
    val dataArray: Array[DTypeToScala[D]] = cpuTensor.toArray.asInstanceOf[Array[DTypeToScala[D]]]
    val shape = cpuTensor.size
    val ndim = shape.length

    // 根据维度创建对应的多维数组
    val ndArray = shape.length match {
      case 0 =>
        // 标量
        dataArray.head.asInstanceOf[DTypeToScala[D]]
      case 1 =>
        // 1D数组
        dataArray
      case 2 =>
        // 2D数组
        dataArray.grouped(shape(1)).toArray
//        reshapeArray[DTypeToScala[D]](dataArray, shape(0), shape(1))
      case 3 =>
        // 3D数组
//        arr.grouped(dim2 * dim3).map(_.grouped(dim3).toArray).toArray
        reshapeArray[DTypeToScala[D]](dataArray, shape(0), shape(1), shape(2))
      case 4 =>
        // 4D数组
        reshapeArray(dataArray, shape(0), shape(1), shape(2), shape(3))
      case 5 =>
        // 5D数组
        reshapeArray(dataArray, shape(0), shape(1), shape(2), shape(3), shape(4))
      case _ =>
        throw new IllegalArgumentException(s"Unsupported tensor dimension: ${shape.length}")
    }
    val numpyDtype: torch.numpy.enums.DType = this.dtype match {
      case DType.float16 | DType.bfloat16 => torch.numpy.enums.DType.Float16
      case DType.float32                  => torch.numpy.enums.DType.Float32
      case DType.float64                  => torch.numpy.enums.DType.Float64
      case DType.int8 | DType.uint8       => torch.numpy.enums.DType.Int8
      case DType.int16                    => torch.numpy.enums.DType.Int16
      case DType.int32                    => torch.numpy.enums.DType.Int32
      case DType.int64                    => torch.numpy.enums.DType.Int64
      case DType.bool                     => torch.numpy.enums.DType.Bool
      case DType.complex64                => torch.numpy.enums.DType.Float64
      case DType.complex128               => torch.numpy.enums.DType.Float64
      case _ => throw new UnsupportedOperationException(s"Unsupported dtype: ${this.dtype}")
    }

//    NDArray(ndArray)
    NDArray(dataArray, shape, ndim, this.toNumpyDType).reshape(shape*)
  }

  def toNumpyDType: torch.numpy.enums.DType = {
    this.dtype match {
      case DType.float16 | DType.bfloat16 => torch.numpy.enums.DType.Float16
      case DType.float32                  => torch.numpy.enums.DType.Float32
      case DType.float64                  => torch.numpy.enums.DType.Float64
      case DType.int8 | DType.uint8       => torch.numpy.enums.DType.Int8
      case DType.int16                    => torch.numpy.enums.DType.Int16
      case DType.int32                    => torch.numpy.enums.DType.Int32
      case DType.int64                    => torch.numpy.enums.DType.Int64
      case DType.bool                     => torch.numpy.enums.DType.Bool
      case DType.complex64                => torch.numpy.enums.DType.Float64
      case DType.complex128               => torch.numpy.enums.DType.Float64
      // 添加其他需要支持的dtype映射
      case _ => throw new UnsupportedOperationException(s"Unsupported dtype: ${this.dtype}")
    }
  }

  /** 将一维数组重塑为二维数组
    */
  private def reshapeArray[T: ClassTag](arr: Array[T], dim1: Int, dim2: Int): Array[Array[T]] = {
    arr.grouped(dim2).toArray
  }

  /** 将一维数组重塑为三维数组
    */
  private def reshapeArray[T: ClassTag](
      arr: Array[T],
      dim1: Int,
      dim2: Int,
      dim3: Int
  ): Array[Array[Array[T]]] = {
    arr.grouped(dim2 * dim3).map(_.grouped(dim3).toArray).toArray
  }

  /** 将一维数组重塑为四维数组
    */
  private def reshapeArray[T: ClassTag](
      arr: Array[T],
      dim1: Int,
      dim2: Int,
      dim3: Int,
      dim4: Int
  ): Array[Array[Array[Array[T]]]] = {
    arr
      .grouped(dim2 * dim3 * dim4)
      .map(
        _.grouped(dim3 * dim4)
          .map(_.grouped(dim4).toArray)
          .toArray
      )
      .toArray
  }

  /** 将一维数组重塑为五维数组
    */
  private def reshapeArray[T: ClassTag](
      arr: Array[T],
      dim1: Int,
      dim2: Int,
      dim3: Int,
      dim4: Int,
      dim5: Int
  ): Array[Array[Array[Array[Array[T]]]]] = {
    arr
      .grouped(dim2 * dim3 * dim4 * dim5)
      .map(
        _.grouped(dim3 * dim4 * dim5)
          .map(
            _.grouped(dim4 * dim5)
              .map(_.grouped(dim5).toArray)
              .toArray
          )
          .toArray
      )
      .toArray
  }
  def toBuffer: TypedBuffer[DTypeToScala[D]] =
    to(device = CPU).native.createBuffer[TypedBuffer[DTypeToScala[D]]]()

  def tolist(un_used: Int*) = this.toArray.toList

  def tolist = this.toArray.toList

  def toList = this.tolist

  def toList(un_used: Int*) = this.toArray.toList

  def toArray: Array[DTypeToScala[D]] =

    val tensor = to(device = CPU)
    def writeArray[A: ClassTag, B <: Buffer](getElem: B => A): Array[A] =
      val a = new Array[A](numel.toInt)
      if numel > 0 then
        val buf = tensor.native.contiguous.createBuffer[B]
        var i = 0
        while i < a.length do
          a(i) = getElem(buf)
          i += 1
      a

    def writeRawArray[A <: ScalaType: ClassTag](
        get: (Array[A], TypedBuffer[A]) => TypedBuffer[A]
    ): Array[A] =
      val a = new Array[A](numel.toInt)
      if numel > 0 then
        val _ = get(a, tensor.native.contiguous.createBuffer[TypedBuffer[A]])
      a

    import ScalarType.*
    val out = tensor.native.dtype().toScalarType.intern() match
      case Byte         => to(dtype = int32).toArray.map(UByte.apply)
      case Char         => writeRawArray[Byte]((a, b) => b.get(a))
      case Short        => writeRawArray[Short]((a, b) => b.get(a))
      case Int          => writeRawArray[Int]((a, b) => b.get(a))
      case Long         => writeRawArray[Long]((a, b) => b.get(a))
      case Float        => writeRawArray[Float]((a, b) => b.get(a))
      case Double       => writeRawArray[Double]((a, b) => b.get(a))
      case Half         => to(dtype = float32).toArray
      case ComplexHalf  => to(dtype = complex64).toArray
      case ComplexFloat => writeArray[Complex[Float], FloatBuffer](b => Complex(b.get(), b.get()))
      case ComplexDouble =>
        writeArray[Complex[Double], DoubleBuffer](b => Complex(b.get(), b.get()))
      case Bool          => writeArray[Boolean, ByteBuffer](b => b.get > 0)
      case QInt8         => ???
      case QUInt8        => ???
      case QInt32        => ???
      case BFloat16      => to(dtype = float32).toArray
      case QUInt4x2      => ???
      case QUInt2x4      => ???
      case Bits1x8       => ???
      case Bits2x4       => ???
      case Bits4x2       => ???
      case Bits8         => ???
      case Bits16        => ???
      case Float8_e5m2   => to(dtype = float32).toArray
      case Float8_e4m3fn => to(dtype = float32).toArray
      case Undefined     => ???
      case NumOptions    => ???
    out.asInstanceOf[Array[DTypeToScala[D]]]

  def toSeq: Seq[DTypeToScala[D]] = ArraySeq.unsafeWrapArray(toArray)

  def _dimV: Long = native._dimV()

  def _dimI: Long = native._dimI()

  def _nnz(): Long = native._nnz()

  def nnz(un_used: Int*): Long = native._nnz()

  def dense_dim(un_used: Int*): Long = native.dense_dim()

  def sparse_dim(un_used: Int*): Long = native.sparse_dim()

  def coalesce(un_used: Int*): Tensor[D] = fromNative(native.coalesce)

  def is_coalesced(un_used: Int*): Boolean = native.is_coalesced()

  def values(un_used: Int*): Tensor[D] = fromNative(native.values)

  def nnz: Long = native._nnz()

  def dense_dim: Long = native.dense_dim()

  def sparse_dim: Long = native.sparse_dim()

  def coalesce: Tensor[D] = fromNative(native.coalesce)

  def is_coalesced: Boolean = native.is_coalesced()

  def values(): Tensor[D] = fromNative(native.values)

  def indices[D2 <: SparseIntNN](): Tensor[D2] = fromNative(native.indices)

  def crow_indices[D2 <: SparseIntNN](): Tensor[D2] = fromNative(native.crow_indices)

  def col_indices[D2 <: SparseIntNN](): Tensor[D2] = fromNative(native.col_indices)

  def ccol_indices[D2 <: SparseIntNN](): Tensor[D2] = fromNative(native.ccol_indices)

  def row_indices[D2 <: SparseIntNN](): Tensor[D2] = fromNative(native.row_indices)

  def to_sparse_csr(un_used: Int*): Tensor[D] = fromNative(native.to_sparse_csr)

  def to_sparse_csc(un_used: Int*): Tensor[D] = fromNative(native.to_sparse_csc)

  def to_sparse_csr: Tensor[D] = fromNative(native.to_sparse_csr)

  def to_sparse_csc: Tensor[D] = fromNative(native.to_sparse_csc)

  def to_sparse_bsr(blockSize: Seq[Long]): Tensor[D] = fromNative(native.to_sparse_bsr(blockSize*))

  def to_sparse_bsc(blockSize: Seq[Long]): Tensor[D] = fromNative(native.to_sparse_bsc(blockSize*))

  def to_sparse_coo(un_used: Int*): Tensor[D] = fromNative(native.to_sparse)

  def to_sparse(un_used: Int*): Tensor[D] = fromNative(native.to_sparse)

  def to_sparse_coo: Tensor[D] = fromNative(native.to_sparse)

  def to_sparse: Tensor[D] = fromNative(native.to_sparse)

  def to_sparse_csr(dense_dim: Int): Tensor[D] = fromNative(
    native.to_sparse_csr(LongOptional(dense_dim.toLong))
  )

  def to_sparse_csc(dense_dim: Int): Tensor[D] = fromNative(
    native.to_sparse_csc(LongOptional(dense_dim.toLong))
  )

  def to_sparse_bsr(blockSize: Seq[Long], dense_dim: Int): Tensor[D] = fromNative(
    native.to_sparse_bsr(blockSize.toArray, LongOptional(dense_dim.toLong))
  )

  def to_sparse_bsc(blockSize: Seq[Long], dense_dim: Int): Tensor[D] = fromNative(
    native.to_sparse_bsc(blockSize.toArray, LongOptional(dense_dim.toLong))
  )

  def to_sparse_coo(sparse_dim: Int): Tensor[D] = fromNative(native.to_sparse(sparse_dim.toLong))

  def to_dense(un_used: Int*): Tensor[D] = fromNative(native.to_dense)

  def to_mkldnn(un_used: Int*): Tensor[D] = fromNative(native.to_mkldnn)

  def is_sparse(un_used: Int*): Boolean = {
    val sparseLayout = Array(Sparse, SparseCsc, SparseBsc, SparseBsr, SparseCsr)
    if sparseLayout.contains(fromNative(native).layout) then true else false
  }

  def to_dense: Tensor[D] = fromNative(native.to_dense)

  def to_mkldnn: Tensor[D] = fromNative(native.to_mkldnn)

  def is_sparse: Boolean = {
    val sparseLayout = Array(Sparse, SparseCsc, SparseBsc, SparseBsr, SparseCsr)
    if sparseLayout.contains(fromNative(native).layout) then true else false
  }

  def __dispatch_requires_grad_(): this.type = {
    native.__dispatch_requires_grad_()
    this
  }

  def put[D2 <: DType](x: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(native.put(x.native))

  def toBackend(b: Int): Tensor[D] = fromNative(native.toBackend(b))

  def not(un_used: Int*): Tensor[D] = fromNative(native.not())

  def subtract(un_used: Int*): Tensor[D] = fromNative(native.subtract())

  def not: Tensor[D] = fromNative(native.not())

  def subtract: Tensor[D] = fromNative(native.subtract())

  def addPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.addPut(other.native)
  )

  def subtractPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.subtractPut(other.native)
  )

  def multiplyPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.multiplyPut(other.native)
  )

  def dividePut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.dividePut(other.native)
  )

  def addPut[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.addPut(toScalar(other))
  )

  def subtractPut[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.subtractPut(toScalar(other))
  )

  def multiplyPut[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.multiplyPut(toScalar(other))
  )

  def dividePut[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.dividePut(toScalar(other))
  )

  def andPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.andPut(other.native)
  )

  def orPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.orPut(other.native)
  )

  def xorPut[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.xorPut(other.native)
  )

  def get(index: Tensor[Int64] | Tensor[Int32]): Tensor[D] = {
    index.dtype match
      case torch.int64 => fromNative(native.get(index.native))
      case torch.int32 => fromNative(native.get(index.to(dtype = torch.int64).native))
  }

  def get(index: Int): Tensor[D] = fromNative(native.get(index.toLong))

  def get[S <: ScalaType](index: S): Tensor[D] = fromNative(native.get(toScalar(index)))

  def mutable_grad(un_used: Int*): Tensor[D] = fromNative(native.mutable_grad())

  def mutable_grad: Tensor[D] = fromNative(native.mutable_grad())

  def conj_physical(un_used: Int*): Tensor[D] = fromNative(native.conj_physical())

  def conj_physical: Tensor[D] = fromNative(native.conj_physical())

  def conj_physical_(): this.type = {
    native.conj_physical_()
    this
  }

  def resolve_conj(unused: Int*): Tensor[D] = fromNative(native.resolve_conj())

  def resolve_neg(unused: Int*): Tensor[D] = fromNative(native.resolve_neg())

  def resolve_conj: Tensor[D] = fromNative(native.resolve_conj())

  def resolve_neg: Tensor[D] = fromNative(native.resolve_neg())

  //  def rename_():
//  def align_to(other: Tensor[D]): Tensor[D] = fromNative(native.align_to(other.native))
  def align_as[D2 <: DType](other: Tensor[D2]): Tensor[Promoted[D, D2]] = fromNative(
    native.align_as(other.native)
  )

  def angle(un_used: Int*): Tensor[D] = fromNative(native.angle())

  def chalf(unused: Int*): Tensor[D] = fromNative(native.chalf())

  def absolute(unused: Int*): Tensor[D] = fromNative(native.absolute())

  def sgn(unused: Int*): Tensor[D] = fromNative(native.sgn())

  def angle: Tensor[D] = fromNative(native.angle())

  def chalf: Tensor[D] = fromNative(native.chalf())

  def absolute: Tensor[D] = fromNative(native.absolute())

  def sgn: Tensor[D] = fromNative(native.sgn())

  def abs_(): Tensor[D] = fromNative(native.abs_())

  def absolute_(): this.type = {
    native.absolute_()
    this
  }

  def sgn_(): this.type = {
    native.sgn_()
    this
  }

  def argmax(unused: Int*): Tensor[D] = fromNative(native.argmax())

  def argmin(unused: Int*): Tensor[D] = fromNative(native.argmin())

  def argmax: Tensor[D] = fromNative(native.argmax())

  def argmin: Tensor[D] = fromNative(native.argmin())

  def argmin(dim: Int, keepdim: Boolean = false): Tensor[D] = fromNative(
    native.argmin(new LongOptional(dim.toLong), keepdim)
  )

  def argmax_dimInt(dim: Int, keepdim: Boolean = false): Tensor[D] = fromNative(
    native.argmax(new LongOptional(dim.toLong), keepdim)
  )

  def all(dim: Seq[Int], keepdim: Boolean): Tensor[D] = fromNative(
    native.all(dim.map(_.toLong).toArray, keepdim)
  )

  def all(dim: Seq[Int]): Tensor[D] = fromNative(native.all(dim.map(_.toLong)*))

  def all(dim: Int, keepdim: Boolean = false): Tensor[D] = fromNative(
    native.all(dim.toLong, keepdim)
  )

  def all(dim: Int): Tensor[D] = fromNative(native.all(dim.toLong))
  def any(dim: Seq[Int], keepdim: Boolean): Tensor[D] = fromNative(
    native.any(dim.map(_.toLong).toArray, keepdim)
  )

  def any(dim: Seq[Int]): Tensor[D] = fromNative(native.any(dim.map(_.toLong)*))

//  def any(dim: Int, keepdim: Boolean = false): Tensor[D] = fromNative(native.any(dim.toLong,keepdim))
  def any(dim: Int): Tensor[D] = fromNative(native.any(dim.toLong))

//  def acos(): Tensor[D] = fromNative(native.acos())

  def arccos(un_used: Int*): Tensor[D] = fromNative(native.arccos())

  def arccos: Tensor[D] = fromNative(native.arccos())

  //  def add(other: Tensor[D]): Tensor[D] = fromNative(native.add(other.native))

  def addmv[D1 <: DType, D2 <: DType](mat: Tensor[D1], vec: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(native.addmv(mat.native, vec.native))

  def addmv[D1 <: DType, D2 <: DType, S <: ScalaType](
      mat: Tensor[D1],
      vec: Tensor[D2],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D2]] =
    fromNative(native.addmv(mat.native, vec.native, toScalar(beta), toScalar(alpha)))

  def addr[D1 <: DType, D2 <: DType](vec1: Tensor[D1], vec2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
    fromNative(native.addr(vec1.native, vec2.native))

//  def addr[D1 <: DType, D2 <: DType, S <: ScalaType](vec1: Tensor[D1], vec2: Tensor[D2]): Tensor[Promoted[D1, D2]] =
//    fromNative(native.addr(vec1.native, vec2.native))

  def addr[D1 <: DType, D2 <: DType, S <: ScalaType](
      vec1: Tensor[D1],
      vec2: Tensor[D2],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D2]] =
    fromNative(native.addr(vec1.native, vec2.native, toScalar(beta), toScalar(alpha)))

  def acos_(): this.type = {
    native.acos_()
    this
  }

  def arccos_(): this.type = {
    native.arccos_()
    this
  }

  def add_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.add_(other.native)
    this
  }

  def addmv_[D1 <: DType, D2 <: DType](mat: Tensor[D1], vec: Tensor[D2]): this.type = {
    native.addmv_(mat.native, vec.native)
    this
  }

  def addmv_[D1 <: DType, D2 <: DType, S <: ScalaType](
      mat: Tensor[D1],
      vec: Tensor[D2],
      beta: S,
      alpha: S
  ): this.type = {
    native.addmv_(mat.native, vec.native, toScalar(beta), toScalar(alpha))
    this
  }

  def addr_[D1 <: DType, D2 <: DType, S <: ScalaType](
      vec1: Tensor[D1],
      vec2: Tensor[D2],
      beta: S,
      alpha: S
  ): this.type = {
    native.addr_(vec1.native, vec2.native, toScalar(beta), toScalar(alpha))
    this
  }

  def addr_[D1 <: DType, D2 <: DType](vec1: Tensor[D1], vec2: Tensor[D2]): this.type = {
    native.addr_(vec1.native, vec2.native)
    this
  }

  def acosh(un_used: Int*): Tensor[D] = fromNative(native.acosh())

  def arccosh(un_used: Int*): Tensor[D] = fromNative(native.arccosh())

  def asinh(un_used: Int*): Tensor[D] = fromNative(native.asinh())

  def arcsinh(un_used: Int*): Tensor[D] = fromNative(native.arcsinh())

  def atanh(un_used: Int*): Tensor[D] = fromNative(native.atanh())

  def arctanh(un_used: Int*): Tensor[D] = fromNative(native.arctanh())

  def acosh: Tensor[D] = fromNative(native.acosh())

  def arccosh: Tensor[D] = fromNative(native.arccosh())

  def asinh: Tensor[D] = fromNative(native.asinh())

  def arcsinh: Tensor[D] = fromNative(native.arcsinh())

  def atanh: Tensor[D] = fromNative(native.atanh())

  def arctanh: Tensor[D] = fromNative(native.arctanh())

  def as_strided(li: Seq[Int], li2: Seq[Int]): Tensor[D] = fromNative(
    native.as_strided(li.map(_.toLong).toArray, li2.map(_.toLong)*)
  )

  def acosh_(): this.type = {
    native.acosh_()
    this
  }

  def arccosh_(): this.type = {
    native.arccosh_()
    this
  }

  def asinh_(): this.type = {
    native.asinh_()
    this
  }

  def arcsinh_(): this.type = {
    native.arcsinh_()
    this
  }

  def atanh_(): this.type = {
    native.atanh_()
    this
  }

  def arctanh_(): this.type = {
    native.arctanh_()
    this
  }

  def as_strided_(li: Seq[Int], li2: Seq[Int]): this.type = {
    native.as_strided_(li.map(_.toLong).toArray, li2.map(_.toLong)*)
    this
  }

  def asin(un_used: Int*): Tensor[D] = fromNative(native.asin())

  def arcsin(un_used: Int*): Tensor[D] = fromNative(native.arcsin())

  def atan(un_used: Int*): Tensor[D] = fromNative(native.atan())

  def arctan(un_used: Int*): Tensor[D] = fromNative(native.arctan())

  def asin: Tensor[D] = fromNative(native.asin())

  def arcsin: Tensor[D] = fromNative(native.arcsin())

  def atan: Tensor[D] = fromNative(native.atan())

  def arctan: Tensor[D] = fromNative(native.arctan())

  def asin_(): this.type = {
    native.asin_()
    this
  }

  def arcsin_(): this.type = {
    native.arcsin_()
    this
  }

  def atan_(): this.type = {
    native.atan_()
    this
  }

  def arctan_(): this.type = {
    native.arctan_()
    this
  }

  def index_put(
      indices: Seq[Tensor[Int64]],
      value: Tensor[D],
      accumulate: Boolean = false
  ): this.type = {
    val list = new TensorOptionalList()
    indices.map(tensor => list.push_back(new TensorOptional(tensor.native)))
    native.index_put(list, value.native, accumulate)
    this
  }
  def index_put_(indices: Seq[Tensor[Int64]], value: Tensor[D]): this.type = {
    val list = new TensorOptionalList()
    indices.map(tensor => list.push_back(new TensorOptional(tensor.native)))
    native.index_put_(list, value.native)
    this
  }

  def where[S <: ScalaType](
      condition: Tensor[Bool],
      other: S
  ): Tensor[Promoted[Int64, ScalaToDType[S]]] =
    fromNative(native.where(condition.native, toScalar(other)))

  def where(condition: Tensor[Bool], other: Tensor[D]): Tensor[Int64] =
    fromNative(native.where(condition.native, other.native))

//  def index_put[S <: ScalaType](indices: Seq[Tensor[Int64]], value: S, acc: Boolean = false): this.type = {
//    val list = new TensorIndexVector()
//    indices.map(tensor => list.push_back(new TensorIndex(tensor.native)))
//    native.index_put(list, toScalar(value), acc)
//    this
//  }

  def index_put_[S <: ScalaType](indices: Seq[Tensor[Int64]], value: S): this.type = {
    val list = new TensorIndexVector()
    indices.map(tensor => list.push_back(new TensorIndex(tensor.native)))
    native.index_put_(list, toScalar(value))
    this
  }

  def bincount[D1 <: DType](weights: Tensor[D1], minlength: Int = 0): Tensor[Promoted[D, D1]] =
    fromNative(native.bincount(new TensorOptional(weights.native), minlength.toLong))

  def bincount(): Tensor[D] = fromNative(native.bincount())

  def baddbmm[D1 <: DType](batch1: Tensor[D1], batch2: Tensor[D1]): Tensor[Promoted[D1, D]] =
    fromNative(native.baddbmm(batch1.native, batch2.native))

  def baddbmm[D1 <: DType, S <: ScalaType](
      batch1: Tensor[D1],
      batch2: Tensor[D1],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D]] =
    fromNative(native.baddbmm(batch1.native, batch2.native, toScalar(beta), toScalar(alpha)))
  def bernoulli(p: Double): Tensor[D] = fromNative(native.bernoulli(p))

  def bernoulli(un_used: Int*): Tensor[D] = fromNative(native.bernoulli())

  def bitwise_not(un_used: Int*): Tensor[D] = fromNative(native.bitwise_not())

  def bernoulli: Tensor[D] = fromNative(native.bernoulli())

  def bitwise_not: Tensor[D] = fromNative(native.bitwise_not())

  def baddbmm_[D1 <: DType](batch1: Tensor[D1], batch2: Tensor[D1]): this.type = {
    native.baddbmm_(batch1.native, batch2.native)
    this
  }

  def baddbmm_[D1 <: DType, S <: ScalaType](
      batch1: Tensor[D1],
      batch2: Tensor[D1],
      beta: S,
      alpha: S
  ): this.type = {
    native.baddbmm_(batch1.native, batch2.native, toScalar(beta), toScalar(alpha))
    this
  }

  def bernoulli_[D1 <: DType](p: Tensor[D1]): this.type = {
    native.bernoulli_(p.native)
    this
  }

  def bernoulli_(): this.type = {
    native.bernoulli_()
    this
  }

  def bitwise_not_(): this.type = {
    native.bitwise_not_()
    this
  }

  def copysign[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.copysign(other.native)
  )

  def copysign_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.copysign_(other.native)
    this
  }

  def copysign[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.copysign(toScalar(other))
  )

  def copysign_[S <: ScalaType](other: S): this.type = {
    native.copysign_(toScalar(other))
    this
  }

  def bmm[D1 <: DType](mat2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bmm(mat2.native)
  )

  def broadcast_to(size: Seq[Int]): Tensor[D] = fromNative(native.broadcast_to(size.map(_.toLong)*))

  def _lazy_clone(): Tensor[D] = fromNative(native._lazy_clone())

  def tensor_split(tensor_indices_or_sections: Tensor[Int64], dim: Int = 0): Seq[Tensor[D]] = {
    val res = native.tensor_split(tensor_indices_or_sections.native, dim.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq

  }
  def tensor_split(tensor_indices_or_sections: Tensor[Int64]): Seq[Tensor[D]] = {
    val res = native.tensor_split(tensor_indices_or_sections.native)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq

  }
  def tensor_split_with_dim(indices: Seq[Int], dim: Int = 0): Seq[Tensor[D]] = {
    val res = native.tensor_split(indices.map(_.toLong).toArray, dim.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def tensor_split(indices: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.tensor_split(indices.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def tensor_split(sections: Int): Seq[Tensor[D]] = {
    val res = native.tensor_split(sections.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def tensor_split(sections: Int, dim: Int): Seq[Tensor[D]] = {
    val res = native.tensor_split(sections.toLong, dim.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def logical_xor[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_xor(other.native)
  )

  def logical_or[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_or(other.native)
  )

  def logical_and[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_and(other.native)
  )

//  def logical_not[D1 <: DType](): Tensor[Promoted[D1, D]] = fromNative(
//    native.logical_not()
//  )

  def ^^[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_xor(other.native)
  )

  def ||[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_or(other.native)
  )

  def &&[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logical_and(other.native)
  )

  def logical_xor_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_xor_(other.native)
    this
  }

  def logical_or_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_or_(other.native)
    this
  }

  def logical_and_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_and_(other.native)
    this
  }

  def logical_not_[D1 <: DType](): this.type = {
    native.logical_not_()
    this
  }

  def ^^=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_xor_(other.native)
    this
  }

  def ||=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_or_(other.native)
    this
  }

  def &&=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.logical_and_(other.native)
    this
  }

  def ~~=[D1 <: DType](): this.type = {
    native.logical_not_()
    this
  }

  def ceil(): Tensor[D] = fromNative(native.ceil())
  def ceil_(): this.type = {
    native.ceil_()
    this
  }

  def clamp(min: Float): Tensor[D] = fromNative(native.clamp(new ScalarOptional(toScalar(min))))

  def clamp_max[D1 <: DType](max: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.clamp_max(max.native)
  )

  def clamp_min[D1 <: DType](min: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.clamp_min(min.native)
  )

  def clamp_max[S <: ScalaType](max: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.clamp_max(toScalar(max))
  )

  def clamp_min[S <: ScalaType](min: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.clamp_min(toScalar(min))
  )

  def clip(min: Float): Tensor[D] = fromNative(native.clip(new ScalarOptional(toScalar(min))))

//  def clip[S <: ScalaType](min: S): Tensor[D] = fromNative(native.clip(new ScalarOptional(toScalar(min))))

  def clamp_(min: Float): this.type = {
    native.clamp_(new ScalarOptional(toScalar(min)))
    this
  }

  def clamp_[S <: ScalaType](min: S): this.type = {
    native.clamp_(new ScalarOptional(toScalar(min)))
    this
  }

  def clamp_(min: Float, max: Float): this.type = {
    native.clamp_(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
    this
  }

  def clamp_[S <: ScalaType](min: S, max: S): this.type = {
    native.clamp_(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
    this
  }

  def clamp_(): this.type = {
    native.clamp_()
    this
  }

  def clamp_max_[S <: ScalaType](max: S): this.type = {
    native.clamp_max_(toScalar(max))
    this
  }

  def clamp_min_[S <: ScalaType](min: S): this.type = {
    native.clamp_min_(toScalar(min))
    this
  }

  def clamp_max_[D1 <: DType](max: Tensor[D1]): this.type = {
    native.clamp_max_(max.native)
    this
  }

  def clamp_min_[D1 <: DType](min: Tensor[D1]): this.type = {
    native.clamp_min_(min.native)
    this
  }

  def clip_(): this.type = {
    native.clip_()
    this
  }

  def clip_[S <: ScalaType](min: S): this.type = {
    native.clip_(new ScalarOptional(toScalar(min)))
    this
  }

  def clip_(min: Float, max: Float): this.type = {
    native.clip_(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
    this
  }

  def clip_[S <: ScalaType](min: S, max: S): this.type = {
    native.clip_(new ScalarOptional(toScalar(min)), new ScalarOptional(toScalar(max)))
    this
  }

//  def clip_[S <: ScalaType](min: S): this.type = {
//    native.clip_(new ScalarOptional(toScalar(min)))
//    this
//  }

  def clip_(min: Float): this.type = {
    native.clip_(new ScalarOptional(toScalar(min)))
    this
  }

  def copy_[D1 <: DType](src: Tensor[D1]): this.type = {
    native.copy_(src.native)
    this
  }

  //  def copy_(src: Tensor[D], non_blocking: Boolean = false) :Tensor[D] = fromNative(native.copy_(src.native,non_blocking))

  def count_nonzero(dim: Seq[Int]): Tensor[D] = fromNative(native.count_nonzero(dim.map(_.toLong)*))

//  def count_nonzero(un_used: Int*): Tensor[D] = fromNative(native.count_nonzero())

//  def corrcoef(un_used: Int*): Tensor[D] = fromNative(native.corrcoef())

  def cov(un_used: Int*): Tensor[D] = fromNative(native.cov())

  def count_nonzero: Tensor[D] = fromNative(native.count_nonzero())

  //  def corrcoef(): Tensor[D] = fromNative(native.corrcoef())

  def cov: Tensor[D] = fromNative(native.cov())

  def cos(un_used: Int*): Tensor[D] = fromNative(native.cos())

  def cosh(un_used: Int*): Tensor[D] = fromNative(native.cosh())

  def cosh: Tensor[D] = fromNative(native.cosh())

  def cos_(): this.type = {
    native.cos_()
    this
  }

  def cosh_(): this.type = {
    native.cosh_()
    this
  }

  def cummax[D1 <: DType, D2 <: DType](dim: Int): (Tensor[D1], Tensor[D2]) = {
    val res = native.cummax(dim.toLong)
    val c = fromNative[D1](res.get0())
    val m = fromNative[D2](res.get1())
    (c, m)
  }

  def cummin[D1 <: DType, D2 <: DType](dim: Int): (Tensor[D1], Tensor[D2]) = {
    val res = native.cummin(dim.toLong)
    val c = fromNative[D1](res.get0())
    val m = fromNative[D2](res.get1())
    (c, m)
  }

  def cumprod(dim: Int): Tensor[D] = fromNative(native.cumprod(dim.toLong))

  def cumsum(dim: Int): Tensor[D] = fromNative(native.cumsum(dim.toLong))

  def cumprod_(dim: Int): this.type = {
    native.cumprod_(dim.toLong)
    this
  }

  def cumsum_(dim: Int): this.type = {
    native.cumsum_(dim.toLong)
    this
  }

  def diag_embed(offset: Int = 0, dim1: Int = 0, dim2: Int = 1): Tensor[D] = fromNative(
    native.diag_embed(offset.toLong, dim1.toLong, dim2.toLong)
  )
  def diag_embed: Tensor[D] = fromNative(native.diag_embed())

  def diagflat(offset: Int = 0): Tensor[D] = fromNative(native.diagflat(offset.toLong))

  def diagflat: Tensor[D] = fromNative(native.diagflat())
  def fill_diagonal_(fill_value: Float): this.type = {
    native.fill_diagonal_(toScalar(fill_value))
    this
  }

  def fill_diagonal_[S <: ScalaType](fill_value: S): this.type = {
    native.fill_diagonal_(toScalar(fill_value))
    this
  }

  def diagonal: Tensor[D] = fromNative(native.diagonal())
  def diagonal(offset: Int = 0, dim1: Int = 0, dim2: Int = 1): Tensor[D] = fromNative(
    native.diagonal(offset.toLong, dim1.toLong, dim2.toLong)
  )

  def diff: Tensor[D] = fromNative(native.diff())

//  def dot(other: Tensor[D]): Tensor[D] = fromNative(native.dot(other.native))

//  def vdot(other: Tensor[D]): Tensor[D] = fromNative(native.vdot(other.native))

//  def div(other: Tensor[D]): Tensor[D] = fromNative(native.div(other.native))

  def divide[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.divide(other.native)
  )

  def true_divide[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.true_divide(other.native)
  )

  def div_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.div_(other.native)
    this
  }

  def divide_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.divide_(other.native)
    this
  }

  def true_divide_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.true_divide_(other.native)
    this
  }

  def divide[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.divide(toScalar(other))
  )

  def true_divide[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.true_divide(toScalar(other))
  )

  def div_[S <: ScalaType](other: S): this.type = {
    native.div_(toScalar(other))
    this
  }

  def divide_[S <: ScalaType](other: S): this.type = {
    native.divide_(toScalar(other))
    this
  }

  def true_divide_[S <: ScalaType](other: S): this.type = {
    native.true_divide_(toScalar(other))
    this
  }

  def resize_(size: Seq[Int]): this.type = {
    native.resize_()
    this
  }

//  def new_empty(size: Seq[Int]): Tensor[D] = fromNative(native.new_empty(size.map(_.toLong).toArray*))

  def new_zeros(size: Seq[Int]): Tensor[D] = fromNative(
    native.new_zeros(size.map(_.toLong).toArray*)
  )

  def new_ones(size: Seq[Int]): Tensor[D] = fromNative(native.new_ones(size.map(_.toLong).toArray*))

  def expand(size: Seq[Int], implicits: Boolean = false): Tensor[D] = fromNative(
    native.expand(size.map(_.toLong).toArray, implicits)
  )

  def expand_with_size(size: Seq[Int]): Tensor[D] = fromNative(native.expand(size.map(_.toLong)*))

  def expand_as[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.expand_as(other.native)
  )

  def unflatten(dim: Int, sizes: Seq[Int]): Tensor[D] = fromNative(
    native.unflatten(dim.toLong, sizes.map(_.toLong)*)
  )

  def erf(un_used: Int*): Tensor[D] = fromNative(native.erf())

  def erfc(un_used: Int*): Tensor[D] = fromNative(native.erfc())

  //  def exp(): Tensor[D] = fromNative(native.exp())

  def exp2(un_used: Int*): Tensor[D] = fromNative(native.exp2())

  def expm1(un_used: Int*): Tensor[D] = fromNative(native.expm1())

  def erf: Tensor[D] = fromNative(native.erf())

  def erfc: Tensor[D] = fromNative(native.erfc())

  def exp: Tensor[D] = fromNative(native.exp())

  def exp2: Tensor[D] = fromNative(native.exp2())

  def expm1: Tensor[D] = fromNative(native.expm1())

  def erf_(): this.type = {
    native.erf_()
    this
  }

  def erfc_(): this.type = {
    native.erfc_()
    this
  }

//  def ^e=(): this.type = {
//    native.exp_()
//    this
//  }
  def exp_(): this.type = {
    native.exp_()
    this
  }

  def exp2_(): this.type = {
    native.exp2_()
    this
  }

  def expm1_(): this.type = {
    native.expm1_()
    this
  }

//  def fill(value: Tensor[D]): Tensor[D] = fromNative(native.fill(value.native))

  def floor(un_used: Int*): Tensor[D] = fromNative(native.floor())

  def frac(un_used: Int*): Tensor[D] = fromNative(native.frac())

  def floor: Tensor[D] = fromNative(native.floor())

  def frac: Tensor[D] = fromNative(native.frac())

  def floor_divide[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.floor_divide(other.native)
  )

  def fill_(value: Tensor[D]): this.type = {
    native.fill_(value.native)
    this
  }

  def fill_[S <: ScalaType](value: S): this.type = {
    native.fill_(toScalar(value))
    this
  }

  def floor_(): this.type = {
    native.floor_()
    this
  }

  def floor_divide_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.floor_divide_(other.native)
    this
  }

  def floor_divide_[S <: ScalaType](other: S): this.type = {
    native.floor_divide_(toScalar(other))
    this
  }

  def frac_(): this.type = {
    native.frac_()
    this
  }

  def gcd_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.gcd_(other.native)
    this
  }

  def gcd[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.gcd(other.native)
  )
  def lcm_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.lcm_(other.native)
    this
  }

  def lcm[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.lcm(other.native)
  )

  def index_copy(dim: Int, index: Tensor[Int64] | Tensor[Int32], source: Tensor[D]): Tensor[D] = {
    index.dtype match
      case torch.int64 => fromNative(native.index_copy(dim.toLong, index.native, source.native))
      case torch.int32 =>
        fromNative(
          native.index_copy(dim.toLong, index.to(dtype = torch.int64).native, source.native)
        )
  }

  def index_copy_(dim: Int, index: Tensor[Int64] | Tensor[Int32], source: Tensor[D]): this.type = {
    index.dtype match
      case torch.int64 => native.index_copy_(dim.toLong, index.native, source.native)
      case torch.int32 =>
        native.index_copy_(dim.toLong, index.to(dtype = torch.int64).native, source.native)
    this
  }

  def isclose[D1 <: DType](
      other: Tensor[D1],
      rtol: Double = 1e-05,
      atol: Double = 1e-08,
      equal_nan: Boolean = false
  ): Tensor[Promoted[D1, D]] = fromNative(native.isclose(other.native, rtol, atol, equal_nan))

  def isclose[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.isclose(other.native)
  )

  def is_distributed(un_used: Int*): Boolean = native.is_distributed()

  def isreal(un_used: Int*): Tensor[D] = fromNative(native.isreal())

  def is_distributed: Boolean = native.is_distributed()

  def isreal: Tensor[D] = fromNative(native.isreal())

  def is_same_size[D1 <: DType](other: Tensor[D1]): Boolean = native.is_same_size(other.native)

  def is_nonzero(un_used: Int*): Boolean = native.is_nonzero()

  def is_nonzero: Boolean = native.is_nonzero()

  def dispatch_is_signed(): Boolean = native.__dispatch_is_signed()

  def dispatch_is_inference(): Boolean = native.__dispatch_is_inference()

  def kthvalue[D1 <: DType](k: Int): (Tensor[D1], Tensor[D1]) = {
    val res = native.kthvalue(k.toLong)
    val r = fromNative[D1](res.get0())
    val s = fromNative[D1](res.get1())
    (r, s)
  }
  def kthvalue[D1 <: DType](
      k: Int,
      dim: Int,
      keepdim: Boolean = false
  ): (Tensor[D1], Tensor[D1]) = {
    val res = native.kthvalue(k.toLong, dim.toLong, keepdim)
    val r = fromNative[D1](res.get0())
    val s = fromNative[D1](res.get1())
    (r, s)
  }

  def kron[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.kron(other.native)
  )
  def nan_to_num_(): this.type = {
    native.nan_to_num_()
    this
  }

  def nan_to_num(unused: Int*): Tensor[D] = fromNative(native.nan_to_num())

  def nan_to_num: Tensor[D] = fromNative(native.nan_to_num())

  def ldexp[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.ldexp(other.native)
  )

//  def log(): Tensor[D] = fromNative(native.log())

  def log10(un_used: Int*): Tensor[D] = fromNative(native.log10())

  def log1p(un_used: Int*): Tensor[D] = fromNative(native.log1p())

  def log2(un_used: Int*): Tensor[D] = fromNative(native.log2())

  def log10: Tensor[D] = fromNative(native.log10())

  def log1p: Tensor[D] = fromNative(native.log1p())

  def log2: Tensor[D] = fromNative(native.log2())

  def ldexp_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.ldexp_(other.native)
    this
  }

  def log_(): this.type = {
    native.log_()
    this
  }

  def log10_(): this.type = {
    native.log10_()
    this
  }

  def log1p_(): this.type = {
    native.log1p_()
    this
  }

  def log2_(): this.type = {
    native.log2_()
    this
  }

  def logaddexp[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logaddexp(other.native)
  )
  def logaddexp2[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.logaddexp2(other.native)
  )
  def xlogy_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.xlogy_(other.native)
    this
  }

  def xlogy[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.xlogy(other.native)
  )

  def xlogy_[S <: ScalaType](other: S): this.type = {
    native.xlogy_(toScalar(other))
    this
  }

  def xlogy[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.xlogy(toScalar(other))
  )

  def log_softmax(dim: Int): Tensor[D] = fromNative(native.log_softmax(dim.toLong))

  def logsumexp(dim: Int): Tensor[D] = fromNative(native.logsumexp(dim.toLong))

  def logsumexp(dim: Seq[Int], keepdim: Boolean = false): Tensor[D] = fromNative(
    native.logsumexp(dim.map(_.toLong).toArray, keepdim)
  )

  def logsumexp(dim: Seq[Int]): Tensor[D] = fromNative(native.logsumexp(dim.map(_.toLong)*))

//  def matmul(other: Tensor[D]): Tensor[D] = fromNative(native.matmul(other.native))

  def matrix_power(n: Int): Tensor[D] = fromNative(native.matrix_power(n.toLong))

  def matrix_exp(un_used: Int*): Tensor[D] = fromNative(native.matrix_exp())

  def matrix_exp: Tensor[D] = fromNative(native.matrix_exp())

  def aminmax(dim: Int, keepdim: Boolean = false): (Tensor[D], Tensor[Int64]) = {
    val res = native.aminmax(new LongOptional(dim.toLong), keepdim)
    val r = fromNative[D](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def max_with_dim(dim: Int): (Tensor[D], Tensor[Int64]) = {
    val res = native.max(dim.toLong)
    val r = fromNative[D](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }

  def max_with_dim(dim: Int, keepdim: Boolean = false): (Tensor[D], Tensor[Int64]) = {
    val res = native.max(dim.toLong, keepdim)
    val r = fromNative[D](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }

  def amax(un_used: Int*): Tensor[D] = fromNative(native.amax())

  def amax: Tensor[D] = fromNative(native.amax())

  def amax(dim: Seq[Int], keepdim: Boolean = false): Tensor[D] = fromNative(
    native.amax(dim.map(_.toLong).toArray, keepdim)
  )

//  def mean_with_dim(dim: Seq[Int],keepdim:Boolean = false): Tensor[D] = fromNative(native.mean(dim.map(_.toLong).toArray,keepdim))

//  def nanmean(dim: Seq[Int],keepdim:Boolean = false): Tensor[D] = fromNative(native.nanmean(dim.map(_.toLong).toArray,keepdim))

  def mean_seq(dim: Seq[Int]): Tensor[D] = fromNative(native.mean(dim.map(_.toLong)*))

//  def nanmean(dim: Seq[Int]): Tensor[D] = fromNative(native.nanmean(dim.map(_.toLong)*))

//  def mean(): Tensor[D] = fromNative(native.mean())

  def nanmean(un_used: Int*): Tensor[D] = fromNative(native.nanmean())

  def median(un_used: Int*): Tensor[D] = fromNative(native.median())

  def nanmedian(un_used: Int*): Tensor[D] = fromNative(native.nanmedian())

  def nanmean: Tensor[D] = fromNative(native.nanmean())

  def median: Tensor[D] = fromNative(native.median())

  def nanmedian: Tensor[D] = fromNative(native.nanmedian())

  def median[D1 <: DType](dim: Int = 1): (Tensor[D1], Tensor[Int64]) = {
    val res = native.median(dim.toLong)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def median_with_dim[D1 <: DType](
      dim: Int = 1,
      keepdim: Boolean = false
  ): (Tensor[D1], Tensor[Int64]) = {
    val res = native.median(dim.toLong, keepdim)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def nanmedian[D1 <: DType](dim: Int, keepdim: Boolean = false): (Tensor[D1], Tensor[Int64]) = {
    val res = native.nanmedian(dim.toLong, keepdim)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def nanmedian[D1 <: DType](dim: Int): (Tensor[D1], Tensor[Int64]) = {
    val res = native.nanmedian(dim.toLong)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }

  def min[D1 <: DType](dim: Int): (Tensor[D1], Tensor[Int64]) = {
    val res = native.min(dim.toLong)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }

  def min[D1 <: DType](dim: Int, keepdim: Boolean = false): (Tensor[D], Tensor[Int64]) = {
    val res = native.min(dim.toLong, keepdim)
    val r = fromNative[D](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def amin: Tensor[D] = fromNative(native.amin())

  def amin(un_used: Int*): Tensor[D] = fromNative(native.amin())

  def amin(dim: Seq[Int], keepdim: Boolean = false): Tensor[D] = fromNative(
    native.amin(dim.map(_.toLong).toArray, keepdim)
  )

  def mm[D1 <: DType](mat2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.mm(mat2.native)
  )

  def mode[D1 <: DType](dim: Int = 1, keepdim: Boolean = false): (Tensor[D1], Tensor[Int64]) = {
    val res = native.mode(dim.toLong, keepdim)
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }
  def mode[D1 <: DType]: (Tensor[D1], Tensor[Int64]) = {
    val res = native.mode()
    val r = fromNative[D1](res.get0())
    val s = fromNative[Int64](res.get1())
    (r, s)
  }

  def mv[D1 <: DType](vec: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(native.mv(vec.native))

//  def mul[S <: ScalaType](other: S): Tensor[D] = fromNative(native.mul(toScalar(other)))

  def multiply[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.multiply(toScalar(other))
  )

  def mvlgamma(p: Long): Tensor[D] = fromNative(native.mvlgamma(p))

  def mul_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.mul_(other.native)
    this
  }

  def multiply_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.multiply_(other.native)
    this
  }

  def mul_[S <: ScalaType](other: S): this.type = {
    native.mul_(toScalar(other))
    this
  }

  def multiply_[S <: ScalaType](other: S): this.type = {
    native.multiply_(toScalar(other))
    this
  }

  def mvlgamma_(p: Long): this.type = {
    native.mvlgamma_(p)
    this
  }

  def narrow_copy(dim: Int, start: Int, length: Int): Tensor[D] = fromNative(
    native.narrow_copy(dim.toLong, start.toLong, length.toLong)
  )
  def narrow(dim: Int, start: Int, length: Int): Tensor[D] = fromNative(
    native.narrow(dim.toLong, start.toLong, length.toLong)
  )
  def narrow(dim: Int, start: Tensor[D], length: Int): Tensor[D] = fromNative(
    native.narrow(dim.toLong, start.native, length.toLong)
  )
  def permute_with_seq(dims: Seq[Int]): Tensor[D] = fromNative(native.permute(dims.map(_.toLong)*))

  def moveaxis(source: Seq[Int], destination: Seq[Int]): Tensor[D] = fromNative(
    native.moveaxis(source.map(_.toLong).toArray, destination.map(_.toLong)*)
  )
  def movedim(source: Int, destination: Int): Tensor[D] = fromNative(
    native.movedim(source.toLong, destination.toLong)
  )

  def numpy_T(un_used: Int*): Tensor[D] = fromNative(native.numpy_T())

  def matrix_H(un_used: Int*): Tensor[D] = fromNative(native.matrix_H())

  //  def adjoint() :Tensor[D] = fromNative(native.adjoint())

  def is_pinned(un_used: Int*): Boolean = native.is_pinned()

  def pin_memory(un_used: Int*): Tensor[D] = fromNative(native.pin_memory())

  def numpy_T: Tensor[D] = fromNative(native.numpy_T())

  def matrix_H: Tensor[D] = fromNative(native.matrix_H())

//  def adjoint() :Tensor[D] = fromNative(native.adjoint())

  def is_pinned: Boolean = native.is_pinned()

  def pin_memory: Tensor[D] = fromNative(native.pin_memory())

  def pinverse(rcond: Double = 1e-15): Tensor[D] = fromNative(native.pinverse(rcond))

  def pinverse(un_used: Int*): Tensor[D] = fromNative(native.pinverse())

  def ravel(un_used: Int*): Tensor[D] = fromNative(native.ravel())

  def rad_deg(un_used: Int*): Tensor[D] = fromNative(native.rad2deg())

  def deg2rad(un_used: Int*): Tensor[D] = fromNative(native.deg2rad())

  def reciprocal(un_used: Int*): Tensor[D] = fromNative(native.reciprocal())

//  def neg(): Tensor[D] = fromNative(native.neg())

  def negative(un_used: Int*): Tensor[D] = fromNative(native.negative())

  def pinverse: Tensor[D] = fromNative(native.pinverse())

  def ravel: Tensor[D] = fromNative(native.ravel())

  def rad_deg: Tensor[D] = fromNative(native.rad2deg())

  def deg2rad: Tensor[D] = fromNative(native.deg2rad())

  def reciprocal: Tensor[D] = fromNative(native.reciprocal())

  def negative: Tensor[D] = fromNative(native.negative())

  def rad_deg_(): this.type = {
    native.rad2deg_()
    this
  }

  def deg2rad_(): this.type = {
    native.deg2rad_()
    this
  }

  def reciprocal_(): this.type = {
    native.reciprocal_()
    this
  }

  def neg_(): this.type = {
    native.neg_()
    this
  }

  def negative_(): this.type = {
    native.negative_()
    this
  }

  def repeat_with_seq(repeats: Seq[Int]): Tensor[D] = fromNative(
    native.repeat(repeats.map(_.toLong)*)
  )

  def repeat_interleave[D1 <: DType](repeats: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.repeat_interleave(repeats.native)
  )

  def repeat_interleave[D1 <: DType](
      repeats: Tensor[Int64],
      dim: Option[Long],
      output_size: Option[Long]
  ): Tensor[D1] = {
    val dimOpt = if dim.isDefined then new LongOptional(dim.get) else new LongOptional()
    val outputSizeOpt =
      if output_size.isDefined then new LongOptional(output_size.get) else new LongOptional()
    fromNative(native.repeat_interleave(repeats.native, dimOpt, outputSizeOpt))
  }

  def repeat_interleave[D1 <: DType](repeats: Long, dim: Int): Tensor[D1] =
    fromNative(native.repeat_interleave(repeats, new LongOptional(dim.toLong), new LongOptional()))

  def repeat_interleave[D1 <: DType](
      repeats: Long,
      dim: Option[Long],
      output_size: Option[Long] = None
  ): Tensor[D1] = {
    val dimOpt = if dim.isDefined then new LongOptional(dim.get) else new LongOptional()
    val outputSizeOpt =
      if output_size.isDefined then new LongOptional(output_size.get) else new LongOptional()
    fromNative(native.repeat_interleave(repeats, dimOpt, outputSizeOpt))
  }

  def reshape_as[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.reshape_as(other.native)
  )
  def round_(): this.type = {
    native.round_()
    this
  }

  def round(decimals: Long): Tensor[D] = fromNative(native.round(decimals))

  def round_(decimals: Long): this.type = {
    native.round_(decimals)
    this
  }

  def relu_(): this.type = {
    native.relu_()
    this
  }

  def relu(un_used: Int*): Tensor[D] = fromNative(native.relu())

  def round(un_used: Int*): Tensor[D] = fromNative(native.round())

  def hardshrink(un_used: Int*): Tensor[D] = fromNative(native.hardshrink())

  def relu: Tensor[D] = fromNative(native.relu())

  def round: Tensor[D] = fromNative(native.round())

  def hardshrink: Tensor[D] = fromNative(native.hardshrink())

  def hardshrink[S <: ScalaType](lambda: S = 0.5): Tensor[D] = fromNative(
    native.hardshrink(toScalar(lambda))
  )

  def hardshrink_backward[S <: ScalaType, D1 <: DType](
      grad_out: Tensor[D1],
      lambda: S = 0.5
  ): Tensor[D] = fromNative(native.hardshrink_backward(grad_out.native, toScalar(lambda)))

  def prelu[D1 <: DType](weight: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.prelu(weight.native)
  )

  def rsqrt(un_used: Int*): Tensor[D] = fromNative(native.rsqrt())

  def sigmoid(un_used: Int*): Tensor[D] = fromNative(native.sigmoid())

  def logit(un_used: Int*): Tensor[D] = fromNative(native.logit())

  def rsqrt: Tensor[D] = fromNative(native.rsqrt())

  def sigmoid: Tensor[D] = fromNative(native.sigmoid())

  def logit: Tensor[D] = fromNative(native.logit())

  def logit(eps: Double): Tensor[D] = fromNative(native.logit(new DoubleOptional(eps)))

  def rsqrt_(): this.type = {
    native.rsqrt_()
    this
  }

  def sigmoid_(): this.type = {
    native.sigmoid_()
    this
  }

  def logit_(): this.type = {
    native.logit_()
    this
  }

  def logit_(eps: Double): this.type = {
    native.logit_(new DoubleOptional(eps))
    this
  }

  def slice(dim: Int, start: Int, end: Int, step: Int): Tensor[D] = fromNative(
    native.slice(
      dim.toLong,
      new LongOptional(start.toLong),
      new LongOptional(end.toLong),
      step.toLong
    )
  )

  def slice(
      dim: Int,
      start: Option[Int] = None,
      end: Option[Int] = None,
      step: Int = 1
  ): Tensor[D] = {

    val startOption =
      if start.isDefined then new LongOptional(start.get.toLong) else new LongOptional()
    val endOption = if end.isDefined then new LongOptional(end.get.toLong) else new LongOptional()
    fromNative(native.slice(dim.toLong, startOption, endOption, step.toLong))

  }

  def slice(un_used: Int*): Tensor[D] = fromNative(native.slice())

  def slice: Tensor[D] = fromNative(native.slice())

  def slice_inverse[D1 <: DType](
      src: Tensor[D1],
      dim: Int,
      start: Int,
      end: Int,
      step: Int = 1
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.slice_inverse(
      src.native,
      dim.toLong,
      new LongOptional(start.toLong),
      new LongOptional(end.toLong),
      step.toLong
    )
  )

  def slice_inverse[D1 <: DType](src: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.slice_inverse(src.native)
  )
  def slice_scatter[D1 <: DType](src: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.slice_scatter(src.native)
  )
  def select_scatter[D1 <: DType](src: Tensor[D1], dim: Int, index: Int): Tensor[Promoted[D1, D]] =
    fromNative(native.select_scatter(src.native, dim.toLong, index.toLong))

  def diagonal_scatter[D1 <: DType](
      src: Tensor[D1],
      offset: Int,
      dim1: Int = 0,
      dim2: Int = 1
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.diagonal_scatter(src.native, offset.toLong, dim1.toLong, dim2.toLong)
  )
  def diagonal_scatter[D1 <: DType](src: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.diagonal_scatter(src.native)
  )
  def as_strided_scatter[D1 <: DType](
      src: Tensor[D1],
      size: Seq[Int],
      stride: Seq[Int]
  ): Tensor[Promoted[D1, D]] = {
    fromNative(
      native.as_strided_scatter(src.native, size.map(_.toLong).toArray, stride.map(_.toLong)*)
    )
  }
  def unsafe_split_with_sizes(split_sizes: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.unsafe_split_with_sizes(split_sizes.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def unsafe_split_with_sizes(split_sizes: Seq[Int], dim: Int = 0): Seq[Tensor[D]] = {
    val res = native.unsafe_split_with_sizes(split_sizes.map(_.toLong).toArray, dim.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def split_with_sizes(split_sizes: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.split_with_sizes(split_sizes.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def split_with_sizes(split_sizes: Seq[Int], dim: Int = 0): Seq[Tensor[D]] = {
    val res = native.split_with_sizes(split_sizes.map(_.toLong).toArray, dim.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def hsplit(indices: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.hsplit(indices.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def hsplit(sections: Int): Seq[Tensor[D]] = {
    val res = native.hsplit(sections.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def vsplit(indices: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.vsplit(indices.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def vsplit(sections: Int): Seq[Tensor[D]] = {
    val res = native.vsplit(sections.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def dsplit(indices: Seq[Int]): Seq[Tensor[D]] = {
    val res = native.dsplit(indices.map(_.toLong)*)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def dsplit(sections: Int): Seq[Tensor[D]] = {
    val res = native.dsplit(sections.toLong)
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def smm[D1 <: DType](mat2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.smm(mat2.native)
  )

  def squeeze_(): this.type = {
    native.squeeze_()
    this
  }

  def squeeze(dim: Int): Tensor[D] = fromNative(native.squeeze(dim.toLong))

  def squeeze(dim: Int*): Tensor[D] = fromNative(native.squeeze(dim.map(_.toLong)*))

  def squeeze_(dim: Int): this.type = {
    native.squeeze_(dim.toLong)
    this
  }

  def squeeze_(dim: Int*): this.type = {
    native.squeeze_(dim.map(_.toLong)*)
    this
  }

  def sspaddmm[D1 <: DType, S <: ScalaType](
      mat1: Tensor[D1],
      mat2: Tensor[D1],
      beta: S = 1.0,
      alpha: S = 1.0
  ): Tensor[Promoted[D1, D]] =
    fromNative(native.sspaddmm(mat1.native, mat2.native, toScalar(beta), toScalar(alpha)))

  def sspaddmm[D1 <: DType](mat1: Tensor[D1], mat2: Tensor[D1]): Tensor[Promoted[D1, D]] =
    fromNative(native.sspaddmm(mat1.native, mat2.native))

  def istft(n_fft: Long): Tensor[D] = fromNative(native.istft(n_fft))

  def istft(
      n_fft: Long,
      hop_length: Option[Long] = None,
      win_length: Option[Long] = None,
      window: Option[Tensor[D]] = None,
      center: Boolean = true,
      normalized: Boolean = false,
      onesided: Option[Boolean] = None,
      length: Option[Long] = None,
      return_complex: Boolean = false
  ): Tensor[D] =
    val nativeHopLength =
      if (hop_length.isDefined) new LongOptional(hop_length.get.toLong) else new LongOptional()
    val nativeWinLength =
      if (win_length.isDefined) new LongOptional(win_length.get.toLong) else new LongOptional()
    val nativeWindow =
      if (window.isDefined) new TensorOptional(window.get.native) else new TensorOptional()
    val nativeOnesided =
      if (onesided.isDefined) new BoolOptional(onesided.get) else new BoolOptional()
    val nativeLength =
      if (length.isDefined) new LongOptional(length.get.toLong) else new LongOptional()
    fromNative(
      native.istft(
        n_fft,
        nativeHopLength,
        nativeWinLength,
        nativeWindow,
        center,
        normalized,
        nativeOnesided,
        nativeLength,
        return_complex
      )
    )

  // long n_fft,
  // LongOptional hop_length, LongOptional win_length,  TensorOptional window,
  //  boolean center/*=true*/,  String pad_mode/*="reflect"*/,  boolean normalized/*=false*/,
  //   BoolOptional onesided,  BoolOptional return_complex,
  //  BoolOptional align_to_window
  // torch.stft(input, n_fft, hop_length=None, win_length=None, window=None,
  // center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None)
  def stft(
      n_fft: Long,
      hop_length: Option[Long] = None,
      win_length: Option[Long] = None,
      window: Option[Tensor[D]] = None,
      center: Boolean = true,
      pad_mode: String = "reflect",
      normalized: Boolean = false,
      onesided: Option[Boolean] = None,
      return_complex: Option[Boolean] = None,
      align_to_window: Option[Boolean] = None
  ): Tensor[D] =

    val nativeHopLength =
      if (hop_length.isDefined) new LongOptional(hop_length.get.toLong) else new LongOptional()
    val nativeWinLength =
      if (win_length.isDefined) new LongOptional(win_length.get.toLong) else new LongOptional()
    val nativeWindow =
      if (window.isDefined) new TensorOptional(window.get.native) else new TensorOptional()
    val nativeOnesided =
      if (onesided.isDefined) new BoolOptional(onesided.get) else new BoolOptional()
    val nativeReturnComplex =
      if (return_complex.isDefined) new BoolOptional(return_complex.get) else new BoolOptional()
    val nativeAlignToWindow =
      if (align_to_window.isDefined) new BoolOptional(align_to_window.get) else new BoolOptional()
    fromNative(
      native.stft(
        n_fft,
        nativeHopLength,
        nativeWinLength,
        nativeWindow,
        center,
        pad_mode,
        normalized,
        nativeOnesided,
        nativeReturnComplex,
        nativeAlignToWindow
      )
    )

  //  def sum(): Tensor[D] = fromNative(native.sum())
  def sum(dim: Int*): Tensor[D] = fromNative(native.sum(dim.map(_.toLong)*))

  def sum(dim: Seq[Int], keepdim: Boolean, dtype: ScalarTypeOptional): Tensor[D] = fromNative(
    native.sum(dim.map(_.toLong).toArray, keepdim, dtype)
  )

  def nansum(un_used: Int*): Tensor[D] = fromNative(native.nansum())

  def nansum: Tensor[D] = fromNative(native.nansum())

  def sum_to_size(size: Seq[Int]): Tensor[D] = fromNative(native.sum_to_size(size.map(_.toLong)*))

  def sinc(un_used: Int*): Tensor[D] = fromNative(native.sinc())

  def sinh(un_used: Int*): Tensor[D] = fromNative(native.sinh())

  //  def detach(): Tensor[D] = fromNative(native.detach())

  def sqrt(un_used: Int*): Tensor[D] = fromNative(native.sqrt())

  def sinc: Tensor[D] = fromNative(native.sinc())

  def sinh: Tensor[D] = fromNative(native.sinh())

//  def detach(): Tensor[D] = fromNative(native.detach())

  def sqrt: Tensor[D] = fromNative(native.sqrt())

//  def square(): Tensor[D] = fromNative(native.square())

  def sin(un_used: Int*): Tensor[D] = fromNative(native.sin())

  def sin_(): this.type = {
    native.sin_()
    this
  }

  def sinc_(): this.type = {
    native.sinc_()
    this
  }

  def sinh_(): this.type = {
    native.sinh_()
    this
  }

  def detach_(): this.type = {
    native.detach_()
    this
  }

  def sqrt_(): this.type = {
    native.sqrt_()
    this
  }

  def square_(): this.type = {
    native.square_()
    this
  }

  def std_with_dim(dim: Seq[Int], unbiased: Boolean = false, keepdim: Boolean = false): Tensor[D] =
    fromNative(native.std(dim.map(_.toLong).toArray, unbiased, keepdim))

  def std(dim: Seq[Int], unbias: Boolean = false): Tensor[D] = fromNative(
    native.std(dim.map(_.toLong).toArray, unbias)
  )

  def std(unbiased: Boolean): Tensor[D] = fromNative(native.std(unbiased))

  def prod_with_dim(dim: Int, keepdim: Boolean = false, dtype: ScalarTypeOptional): Tensor[D] =
    fromNative(native.prod(dim.toLong, keepdim, dtype))

  def prod(dim: Int): Tensor[D] = fromNative(native.prod(dim.toLong))

  def prod(un_used: Int*): Tensor[D] = fromNative(native.prod())

  def tan(un_used: Int*): Tensor[D] = fromNative(native.tan())

  def tanh(un_used: Int*): Tensor[D] = fromNative(native.tanh())

  def prod: Tensor[D] = fromNative(native.prod())

  def tan: Tensor[D] = fromNative(native.tan())

  def tanh: Tensor[D] = fromNative(native.tanh())

  def t_(): this.type = {
    native.t_()
    this
  }

  def tan_(): this.type = {
    native.tan_()
    this
  }

  def tanh_(): this.type = {
    native.tanh_()
    this
  }

  def tile(dims: Seq[Int]): Tensor[D] = fromNative(native.tile(dims.map(_.toLong)*))

  def transpose_(dim0: Long, dim1: Long): this.type = {
    native.transpose_(dim0, dim1)
    this
  }

  def flip(dims: Seq[Int]): Tensor[D] = fromNative(native.flip(dims.map(_.toLong)*))

  def fliplr(un_used: Int*): Tensor[D] = fromNative(native.fliplr())

  def flipud(un_used: Int*): Tensor[D] = fromNative(native.flipud())

  def fliplr: Tensor[D] = fromNative(native.fliplr())

  def flipud: Tensor[D] = fromNative(native.flipud())

  def roll(shifts: Seq[Int], dims: Seq[Int]): Tensor[D] = fromNative(
    native.roll(shifts.map(_.toLong).toArray, dims.map(_.toLong)*)
  )
  def roll(shifts: Seq[Int]): Tensor[D] = fromNative(native.roll(shifts.map(_.toLong)*))

  def rot90(k: Int, dims: Seq[Int]): Tensor[D] = fromNative(
    native.rot90(k.toLong, dims.map(_.toLong)*)
  )

  def rot90(un_used: Int*): Tensor[D] = fromNative(native.rot90())

  def rot90: Tensor[D] = fromNative(native.rot90())

  def trunc_(): this.type = {
    native.trunc_()
    this
  }

  def fix_(): this.type = {
    native.fix_()
    this
  }

  def trunc(un_used: Int*): Tensor[D] = fromNative(native.trunc())

  def fix(un_used: Int*): Tensor[D] = fromNative(native.fix())

  def trunc: Tensor[D] = fromNative(native.trunc())

  def fix: Tensor[D] = fromNative(native.fix())

  def type_as[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.type_as(other.native)
  )
  def unsqueeze_(dim: Int): this.type = {
    native.unsqueeze_(dim.toLong)
    this
  }

  def `var`(unbiased: Boolean): Tensor[D] = fromNative(native.`var`(unbiased))

  def `var`(dim: Seq[Int], unbiased: Boolean, keepdim: Boolean): Tensor[D] = fromNative(
    native.`var`(dim.map(_.toLong).toArray, unbiased, keepdim)
  )

  def `var`(dim: Seq[Int], unbiased: Boolean): Tensor[D] = fromNative(
    native.`var`(dim.map(_.toLong).toArray, unbiased)
  )

  def `var`(un_used: Int*): Tensor[D] = fromNative(native.`var`())

  def `var`: Tensor[D] = fromNative(native.`var`())

  def view_as[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.view_as(other.native)
  )

  def frexp(un_used: Int*): (Tensor[D], Tensor[D]) = {
    val fr = native.frexp()
    val f = fromNative[D](fr.get0())
    val r = fromNative[D](fr.get1())
    (f, r)
  }

  def frexp: (Tensor[D], Tensor[D]) = {
    val fr = native.frexp()
    val f = fromNative[D](fr.get0())
    val r = fromNative[D](fr.get1())
    (f, r)
  }

  def resize_as_[D1 <: DType](template: Tensor[D1]): this.type = {
    native.resize_as_(template.native)
    this
  }

  def resize_as_sparse_[D1 <: DType](template: Tensor[D1]): Tensor[D] = fromNative(
    native.resize_as_sparse_(template.native)
  )
  def positive(un_used: Int*): Tensor[D] = fromNative(native.positive())

  def positive: Tensor[D] = fromNative(native.positive())

  //  def sub(other: Tensor[D]): Tensor[D] = fromNative(native.sub(other.native))

  def subtract[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.subtract(other.native)
  )

  def heaviside[D1 <: DType](values: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.heaviside(values.native)
  )

  def addmm[D1 <: DType](mat1: Tensor[D1], mat2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.addmm(mat1.native, mat2.native)
  )

  def addmm[D1 <: DType, S <: ScalaType](
      mat1: Tensor[D1],
      mat2: Tensor[D1],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.addmm(mat1.native, mat2.native, toScalar(beta), toScalar(alpha))
  )

  def sub_[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.sub_(other.native)
  )

  def subtract_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.subtract_(other.native)
    this
  }

  def sub_[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.sub_(toScalar(other))
  )

  def subtract_[S <: ScalaType](other: S): this.type = {
    native.subtract_(toScalar(other))
    this
  }

  def sub_[S <: ScalaType](other: S, alpha: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.sub_(toScalar(other), toScalar(alpha))
  )

  def subtract_[S <: ScalaType](other: S, alpha: S): this.type = {
    native.subtract_(toScalar(other), toScalar(alpha))
    this
  }

  def heaviside_[D1 <: DType](values: Tensor[D1]): this.type = {
    native.heaviside_(values.native)
    this
  }

  def addmm_[D1 <: DType](mat1: Tensor[D1], mat2: Tensor[D1]): this.type = {
    native.addmm_(mat1.native, mat2.native)
    this
  }

  def addmm_[D1 <: DType, S <: ScalaType](
      mat1: Tensor[D1],
      mat2: Tensor[D1],
      beta: S,
      alpha: S
  ): this.type = {
    native.addmm_(mat1.native, mat2.native, toScalar(beta), toScalar(alpha))
    this
  }

  def sparse_resize_(size: Seq[Int], sparse_dim: Int, dense_dim: Int): this.type = {
    native.sparse_resize_(size.map(_.toLong).toArray, sparse_dim.toLong, dense_dim.toLong)
    this
  }

  def sparse_resize_and_clear_(size: Seq[Int], sparse_dim: Int, dense_dim: Int): this.type = {
    native.sparse_resize_and_clear_(size.map(_.toLong).toArray, sparse_dim.toLong, dense_dim.toLong)
    this
  }

  def sparse_mask[D1 <: DType](mask: Tensor[Bool]): Tensor[Promoted[D1, D]] = fromNative(
    native.sparse_mask(mask.native)
  )
//  def to_mkldnn():Tensor[D] = fromNative(native.to_mkldnn())
  def dequantize(un_used: Int*): Tensor[D] = fromNative(native.dequantize())

  def q_scale(un_used: Int*): Double = native.q_scale()

  def q_zero_point(un_used: Int*): Long = native.q_zero_point()

  def q_per_channel_scales(un_used: Int*): Tensor[D] = fromNative(native.q_per_channel_scales())

  def q_per_channel_zero_points(un_used: Int*): Tensor[D] = fromNative(
    native.q_per_channel_zero_points()
  )

  def q_per_channel_axis(un_used: Int*): Long = native.q_per_channel_axis()

  def int_repr(un_used: Int*): Tensor[D] = fromNative(native.int_repr())

  def qscheme(un_used: Int*) = native.qscheme()

  def dequantize: Tensor[D] = fromNative(native.dequantize())

  def q_scale: Double = native.q_scale()

  def q_zero_point: Long = native.q_zero_point()

  def q_per_channel_scales: Tensor[D] = fromNative(native.q_per_channel_scales())

  def q_per_channel_zero_points: Tensor[D] = fromNative(native.q_per_channel_zero_points())

  def q_per_channel_axis: Long = native.q_per_channel_axis()

  def int_repr: Tensor[D] = fromNative(native.int_repr())

  def qscheme = native.qscheme()

  def _autocast_to_full_precision(cuda_enabled: Boolean, cpu_enabled: Boolean): Tensor[D] =
    fromNative(native._autocast_to_full_precision(cuda_enabled, cpu_enabled))

  def _autocast_to_reduced_precision[D1 <: DType, D2 <: DType](
      cuda_enabled: Boolean,
      cpu_enabled: Boolean,
      cuda_dtype: D1,
      cpu_dtype: D2
  ): Tensor[D] =
    fromNative(
      native._autocast_to_reduced_precision(
        cuda_enabled,
        cpu_enabled,
        cuda_dtype.toScalarType,
        cpu_dtype.toScalarType
      )
    )

  def set_(source: Storage): this.type = {
    native.set_(source)
    this
  }

  def set_[D1 <: DType](
      source: Tensor[D1],
      storage_offset: Int,
      size: Seq[Int],
      stride: Seq[Int]
  ): this.type = {
    native.set_(
      source.native,
      storage_offset.toLong,
      size.map(_.toLong).toArray,
      stride.map(_.toLong)*
    )
    this
  }

  def set_[D1 <: DType](source: Tensor[D1], storage_offset: Int, size: Seq[Int]): this.type = {
    native.set_(source.native, storage_offset.toLong, size.map(_.toLong)*)
    this
  }

  def is_set_to[D1 <: DType](tensor: Tensor[D1]): Boolean = native.is_set_to(tensor.native)
  def set_[D1 <: DType](source: Tensor[D1]): this.type = {
    native.set_(source.native)
    this
  }

  def set_(): this.type = {
    native.set_()
    this
  }

  def index_fill_[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      value: S
  ): this.type = {
    native.index_fill_(dim.toLong, index.native, toScalar(value))
    this
  }

  def masked_fill_[D1 <: DType](mask: Tensor[Bool], value: Tensor[D1]): this.type = {
    native.masked_fill_(mask.native, value.native)
    this
  }

  def masked_fill_[D1 <: DType, S <: ScalaType](mask: Tensor[Bool], value: S): this.type = {
    native.masked_fill_(mask.native, toScalar(value))
    this
  }

  def masked_fill[D1 <: DType](mask: Tensor[Bool], value: Tensor[D1]): Tensor[Promoted[D, D1]] =
    fromNative(native.masked_fill(mask.native, value.native))

  def masked_scatter_[D1 <: DType](mask: Tensor[Bool], source: Tensor[D1]): this.type = {
    native.masked_scatter_(mask.native, source.native)
    this
  }

  def put_[D1 <: DType](index: Tensor[Int64] | Tensor[Int32], source: Tensor[D1]): this.type = {
    index.dtype match
      case torch.int64 => native.put_(index.native, source.native)
      case torch.int32 => native.put_(index.to(dtype = torch.int64).native, source.native)
    this
  }

  def index_add_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): this.type = {
    index.dtype match
      case torch.int64 => native.index_add_(dim.toLong, index.native, source.native)
      case torch.int32 =>
        native.index_add_(dim.toLong, index.to(dtype = torch.int64).native, source.native)
    this
  }

  def index_add_[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      alpha: S
  ): this.type = {
    index.dtype match
      case torch.int64 =>
        native.index_add_(dim.toLong, index.native, source.native, toScalar(alpha))
      case torch.int32 =>
        native.index_add_(
          dim.toLong,
          index.to(dtype = torch.int64).native,
          source.native,
          toScalar(alpha)
        )
    this
  }

  def index_reduce_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      reduceMode: String
  ): this.type = {
    index.dtype match
      case torch.int64 => native.index_reduce_(dim.toLong, index.native, source.native, reduceMode)
      case torch.int32 =>
        native.index_reduce_(
          dim.toLong,
          index.to(dtype = torch.int64).native,
          source.native,
          reduceMode
        )
    this
  }

  def index_fill_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): this.type = {
    index.dtype match
      case torch.int64 => native.index_fill_(dim.toLong, index.native, source.native)
      case torch.int32 =>
        native.index_fill_(dim.toLong, index.to(dtype = torch.int64).native, source.native)
    this
  }

  def scatter_[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: S
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_(dim.toLong, index.native, toScalar(source))
      case torch.int32 =>
        native.scatter_(dim.toLong, index.to(dtype = torch.int64).native, toScalar(source))
    this
  }

  def scatter_[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: S,
      reduceMode: String
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_(dim.toLong, index.native, toScalar(source), reduceMode)
      case torch.int32 =>
        native.scatter_(
          dim.toLong,
          index.to(dtype = torch.int64).native,
          toScalar(source),
          reduceMode
        )
    this
  }

  def scatter_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_(dim.toLong, index.native, source.native)
      case torch.int32 =>
        native.scatter_(dim.toLong, index.to(dtype = torch.int64).native, source.native)
    this
  }

  def scatter_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      reduceMode: String
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_(dim.toLong, index.native, source.native, reduceMode)
      case torch.int32 =>
        native.scatter_(dim.toLong, index.to(dtype = torch.int64).native, source.native, reduceMode)
    this
  }

  def scatter_add_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1]
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_add_(dim.toLong, index.native, src.native)
      case torch.int32 =>
        native.scatter_add_(dim.toLong, index.to(dtype = torch.int64).native, src.native)
    this
  }

  def scatter_reduce_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1],
      reduceMode: String
  ): this.type = {
    index.dtype match
      case torch.int64 => native.scatter_reduce_(dim.toLong, index.native, src.native, reduceMode)
      case torch.int32 =>
        native.scatter_reduce_(
          dim.toLong,
          index.to(dtype = torch.int64).native,
          src.native,
          reduceMode
        )
    this
  }

  def scatter_reduce_[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1],
      reduceMode: String,
      include_self: Boolean = true
  ): this.type = {
    index.dtype match
      case torch.int64 =>
        native.scatter_reduce_(dim.toLong, index.native, src.native, reduceMode, include_self)
      case torch.int32 =>
        native.scatter_reduce_(
          dim.toLong,
          index.to(dtype = torch.int64).native,
          src.native,
          reduceMode,
          include_self
        )
    this
  }

  def eq_(other: Tensor[D]): this.type = {
    native.eq_(other.native)
    this
  }

  def eq_[S <: ScalaType](other: S): this.type = {
    native.eq_(toScalar(other))
    this
  }
  // mask D2 Bool
  def masked_scatter[D1 <: DType](mask: Tensor[Bool], source: Tensor[D1]): Tensor[Promoted[D1, D]] =
    fromNative(native.masked_scatter(mask.native, source.native))

  def put[D1 <: DType](
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.put(index.native, source.native))
      case torch.int32 =>
        fromNative(native.put(index.to(dtype = torch.int64).native, source.native))
  }

  def index_add[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.index_add(dim.toLong, index.native, source.native))
      case torch.int32 =>
        fromNative(
          native.index_add(dim.toLong, index.to(dtype = torch.int64).native, source.native)
        )
  }

  def index_add[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      alpha: S
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(native.index_add(dim.toLong, index.native, source.native, toScalar(alpha)))
      case torch.int32 =>
        fromNative(
          native.index_add(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            source.native,
            toScalar(alpha)
          )
        )
  }

  def index_reduce[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      reduceMode: String
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(native.index_reduce(dim.toLong, index.native, source.native, reduceMode))
      case torch.int32 =>
        fromNative(
          native.index_reduce(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            source.native,
            reduceMode
          )
        )
  }

  def index_reduce[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      reduceMode: String,
      include_self: Boolean = true
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(
          native.index_reduce(dim.toLong, index.native, source.native, reduceMode, include_self)
        )
      case torch.int32 =>
        fromNative(
          native.index_reduce(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            source.native,
            reduceMode,
            include_self
          )
        )
  }

  def index_fill[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: S
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.index_fill(dim.toLong, index.native, toScalar(source)))
      case torch.int32 =>
        fromNative(
          native.index_fill(dim.toLong, index.to(dtype = torch.int64).native, toScalar(source))
        )

  }
  def index_fill[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.index_fill(dim.toLong, index.native, source.native))
      case torch.int32 =>
        fromNative(
          native.index_fill(dim.toLong, index.to(dtype = torch.int64).native, source.native)
        )

  }

  def scatter[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: S
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.scatter(dim.toLong, index.native, toScalar(source)))
      case torch.int32 =>
        fromNative(
          native.scatter(dim.toLong, index.to(dtype = torch.int64).native, toScalar(source))
        )
  }

  def scatter[D1 <: DType, S <: ScalaType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: S,
      reduceMode: String
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(native.scatter(dim.toLong, index.native, toScalar(source), reduceMode))
      case torch.int32 =>
        fromNative(
          native.scatter(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            toScalar(source),
            reduceMode
          )
        )
  }
  def scatter[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.scatter(dim.toLong, index.native, source.native))
      case torch.int32 =>
        fromNative(native.scatter(dim.toLong, index.to(dtype = torch.int64).native, source.native))
  }

  def scatter[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      source: Tensor[D1],
      reduceMode: String
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(native.scatter(dim.toLong, index.native, source.native, reduceMode))
      case torch.int32 =>
        fromNative(
          native.scatter(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            source.native,
            reduceMode
          )
        )
  }

  def scatter_add[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 => fromNative(native.scatter_add(dim.toLong, index.native, src.native))
      case torch.int32 =>
        fromNative(native.scatter_add(dim.toLong, index.to(dtype = torch.int64).native, src.native))

  }

  def scatter_reduce[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1],
      reduceMode: String
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(native.scatter_reduce(dim.toLong, index.native, src.native, reduceMode))
      case torch.int32 =>
        fromNative(
          native.scatter_reduce(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            src.native,
            reduceMode
          )
        )
  }

  def scatter_reduce[D1 <: DType](
      dim: Int,
      index: Tensor[Int64] | Tensor[Int32],
      src: Tensor[D1],
      reduceMode: String,
      include_self: Boolean = true
  ): Tensor[Promoted[D1, D]] = {
    index.dtype match
      case torch.int64 =>
        fromNative(
          native.scatter_reduce(dim.toLong, index.native, src.native, reduceMode, include_self)
        )
      case torch.int32 =>
        fromNative(
          native.scatter_reduce(
            dim.toLong,
            index.to(dtype = torch.int64).native,
            src.native,
            reduceMode,
            include_self
          )
        )
  }

  def bitwise_and[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_and(toScalar(other))
  )

  def bitwise_or[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_or(toScalar(other))
  )

  def bitwise_xor[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_xor(toScalar(other))
  )

//  def bitwise_not[S <: ScalaType](): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
//    native.bitwise_not()
//  )

  def ~[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_not()
  )

  def &[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_and(toScalar(other))
  )

  def |[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_or(toScalar(other))
  )

  def ^[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] = fromNative(
    native.bitwise_xor(toScalar(other))
  )

  def bitwise_and_[S <: ScalaType](other: S): this.type = {
    native.bitwise_and_(toScalar(other))
    this
  }

  def bitwise_or_[S <: ScalaType](other: S): this.type = {
    native.bitwise_or_(toScalar(other))
    this
  }

  def bitwise_xor_[S <: ScalaType](other: S): this.type = {
    native.bitwise_xor_(toScalar(other))
    this
  }

//  def bitwise_not_() : this.type = {
//    native.bitwise_not_()
//    this
//  }

  def &=[S <: ScalaType](other: S): this.type = {
    native.bitwise_and_(toScalar(other))
    this
  }

  def |=[S <: ScalaType](other: S): this.type = {
    native.bitwise_or_(toScalar(other))
    this
  }

  def ^=[S <: ScalaType](other: S): this.type = {
    native.bitwise_xor_(toScalar(other))
    this
  }

//  def ~=[S <: ScalaType]: this.type = {
//    native.bitwise_not_()
//    this
//  }

  def bitwise_and[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_and(other.native)
  )

  def bitwise_or[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_or(other.native)
  )

  def bitwise_xor[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_xor(other.native)
  )

//  def bitwise_not(un_used: Int*): Tensor[D] = fromNative(
//    native.bitwise_not()
//  )

  def &[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_and(other.native)
  )

  def |[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_or(other.native)
  )

  def ^[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_xor(other.native)
  )

  def ~[D1 <: DType](): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_not()
  )

  def bitwise_and_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_and_(other.native)
    this
  }

  def bitwise_or_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_or_(other.native)
    this
  }

  def bitwise_xor_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_xor_(other.native)
    this
  }

//  def bitwise_not_[D1 <: DType](other: Tensor[D1]): this.type = {
//    native.bitwise_not_(other.native)
//    this
//  }

  def &=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_and_(other.native)
    this
  }

  def |=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_or_(other.native)
    this
  }

  def ^=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_xor_(other.native)
    this
  }

  def ~=(): this.type = {
    native.bitwise_not_()
    this
  }

  def bitwise_left_shift_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_left_shift_(other.native)
    this
  }

  def bitwise_right_shift_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.bitwise_right_shift_(other.native)
    this
  }

  def bitwise_left_shift_[S <: ScalaType](other: S): this.type = {
    native.bitwise_left_shift_(toScalar(other))
    this
  }

  def bitwise_right_shift_[S <: ScalaType](other: S): this.type = {
    native.bitwise_right_shift_(toScalar(other))
    this
  }

  def bitwise_right_shift[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_right_shift(other.native)
  )

  def bitwise_right_shift[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(
      native.bitwise_right_shift(toScalar(other))
    )

  def bitwise_left_shift[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.bitwise_left_shift(other.native)
  )

  def bitwise_left_shift[S <: ScalaType](other: S): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(
      native.bitwise_left_shift(toScalar(other))
    )

  def tril_(): this.type = {
    native.tril_()
    this
  }

  def triu_(): this.type = {
    native.triu_()
    this
  }

  def tril_(diagonal: Long): this.type = {
    native.tril_(diagonal)
    this
  }

  def triu_(diagonal: Long): this.type = {
    native.triu_(diagonal)
    this
  }

  def digamma_(): this.type = {
    native.digamma_()
    this
  }

  // D1 Int , D2 Float
  def lerp_[D1 <: DType, D2 <: DType](end: Tensor[D1], weight: Tensor[D2]): this.type = {
    native.lerp_(end.native, weight.native)
    this
  }

  def lerp_[D1 <: DType, S <: ScalaType](end: Tensor[D1], weight: S): this.type = {
    native.lerp_(end.native, toScalar(weight))
    this
  }

  def addbmm_[D1 <: DType](batch1: Tensor[D1], batch2: Tensor[D1]): this.type = {
    native.addbmm_(batch1.native, batch2.native)
    this
  }

  def addbmm[D1 <: DType, S <: ScalaType](
      batch1: Tensor[D1],
      batch2: Tensor[D1],
      beta: S,
      alpha: S
  ): Tensor[Promoted[D1, D]] =
    fromNative(native.addbmm(batch1.native, batch2.native, toScalar(beta), toScalar(alpha)))

  def addbmm_[D1 <: DType, S <: ScalaType](
      batch1: Tensor[D1],
      batch2: Tensor[D1],
      beta: S,
      alpha: S
  ): this.type = {
    native.addbmm_(batch1.native, batch2.native, toScalar(beta), toScalar(alpha))
    this
  }

  def addbmm[D1 <: DType](batch1: Tensor[D1], batch2: Tensor[D1]): Tensor[Promoted[D1, D]] =
    fromNative(native.addbmm(batch1.native, batch2.native))
  def random_(to: Long): this.type = {
    native.random_(to)
    this
  }

  def random_(from: Long, to: Long): this.type = {
    native.random_(from, new LongOptional(to))
    this
  }

  def random_(): this.type = {
    native.random_()
    this
  }

  def uniform_(): this.type = {
    native.uniform_()
    this
  }

  def uniform_(from: Double, to: Double, seed: Option[Long] = None): this.type = {
    if seed.isDefined then
      val generator = new Generator()
      generator.set_current_seed(seed.get)
      native.uniform_(from, to, new GeneratorOptional(generator))
    else native.uniform_(from, to, new GeneratorOptional())
    this
  }

  def cauchy_(): this.type = {
    native.cauchy_()
    this
  }

  def cauchy_(median: Double, sigma: Double, seed: Option[Long] = None): this.type = {
    if seed.isDefined then
      val generator = new Generator()
      generator.set_current_seed(seed.get)
      native.cauchy_(median, sigma, new GeneratorOptional(generator))
    else native.cauchy_(median, sigma, new GeneratorOptional())

    this
  }

  def log_normal_(): this.type = {
    native.log_normal_()
    this
  }

  def log_normal_(mean: Double, std: Double, seed: Option[Long] = None): this.type = {
    if seed.isDefined then
      val generator = new Generator()
      generator.set_current_seed(seed.get)
      native.log_normal_(mean, std, new GeneratorOptional(generator))
    else native.log_normal_(mean, std, new GeneratorOptional())
    this
  }

  def exponential_(): this.type = {
    native.exponential_()
    this
  }

  def exponential_(lambd: Double, seed: Option[Long] = None): this.type = {
    if seed.isDefined then
      val generator = new Generator()
      generator.set_current_seed(seed.get)
      native.exponential_(lambd, new GeneratorOptional(generator))
    else native.exponential_(lambd, new GeneratorOptional())
    this
  }

  def geometric_(p: Double): this.type = {
    native.geometric_(p)
    this
  }

  def ne_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.ne_(other.native)
    this
  }

  def not_equal_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.not_equal_(other.native)
    this
  }

  def ge_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.ge_(other.native)
    this
  }

  def greater_equal_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.greater_equal_(other.native)
    this
  }

  def le_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.le_(other.native)
    this
  }

  def less_equal_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.less_equal_(other.native)
    this
  }

  def gt_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.gt_(other.native)
    this
  }

  def greater_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.greater_(other.native)
    this
  }

  def lt_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.lt_(other.native)
    this
  }

  def less_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.less_(other.native)
    this
  }

  def ne[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.ne(other.native)
  )

  def not_equal[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.not_equal(other.native)
  )

  def ge[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.ge(other.native)
  )

  def greater_equal[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.greater_equal(other.native)
  )

  def le[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.le(other.native)
  )

  def less_equal[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.less_equal(other.native)
  )

  def gt[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.gt(other.native)
  )

  def greater[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.greater(other.native)
  )

  def lt[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.lt(other.native)
  )

  def less[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.less(other.native)
  )

  def ne_[S <: ScalaType](other: S): this.type = {
    native.ne_(toScalar(other))
    this
  }

  def not_equal_[S <: ScalaType](other: S): this.type = {
    native.not_equal_(toScalar(other))
    this
  }

  def ge_[S <: ScalaType](other: S): this.type = {
    native.ge_(toScalar(other))
    this
  }

  def greater_equal_[S <: ScalaType](other: S): this.type = {
    native.greater_equal_(toScalar(other))
    this
  }

  def le_[S <: ScalaType](other: S): this.type = {
    native.le_(toScalar(other))
    this
  }

  def less_equal_[S <: ScalaType](other: S): this.type = {
    native.less_equal_(toScalar(other))
    this
  }

  def gt_[S <: ScalaType](other: S): this.type = {
    native.gt_(toScalar(other))
    this
  }

  def greater_[S <: ScalaType](other: S): this.type = {
    native.greater_(toScalar(other))
    this
  }

  def lt_[S <: ScalaType](other: S): this.type = {
    native.lt_(toScalar(other))
    this
  }

  def less_[S <: ScalaType](other: S): this.type = {
    native.less_(toScalar(other))
    this
  }

  def ne[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.ne(toScalar(other))
  )

  def not_equal[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.not_equal(toScalar(other))
  )

//  def ge[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
//    native.ge(toScalar(other))
//  )

  def greater_equal[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.greater_equal(toScalar(other))
  )

//  def le[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
//    native.le(toScalar(other))
//  )

  def less_equal[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.less_equal(toScalar(other))
  )

//  def gt[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
//    native.gt(toScalar(other))
//  )

  def greater[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.greater(toScalar(other))
  )

//  def lt[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
//    native.lt(toScalar(other))
//  )

  def less[S <: ScalaType](other: S): Tensor[Bool] = fromNative(
    native.less(toScalar(other))
  )

  def diag(diagonal: Long = 0): Tensor[D] = fromNative(native.diag(diagonal))

  def triu(diagonal: Long = 0): Tensor[D] = fromNative(native.triu(diagonal))

  def diag(un_used: Int*): Tensor[D] = fromNative(native.diag())

  def triu(un_used: Int*): Tensor[D] = fromNative(native.triu())

  def diag: Tensor[D] = fromNative(native.diag())

  def triu: Tensor[D] = fromNative(native.triu())

  def cross[D1 <: DType](other: Tensor[D1], dim: Int): Tensor[Promoted[D1, D]] = fromNative(
    native.cross(other.native, new LongOptional(dim.toLong))
  )
  def cross[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.cross(other.native)
  )
//  def take(index: Tensor[Int64]):Tensor[D] = fromNative(native.take(index.native))

  def take_along_dim(indices: Tensor[Int64] | Tensor[Int32]): Tensor[D] = {
    indices.dtype match
      case torch.int64 => fromNative(native.take_along_dim(indices.native))
      case torch.int32 => fromNative(native.take_along_dim(indices.to(dtype = torch.int64).native))
  }
  def index_select(dim: Int, index: Tensor[Int64] | Tensor[Int32]): Tensor[D] = {
    index.dtype match
      case torch.int64 => fromNative(native.index_select(dim.toLong, index.native))
      case torch.int32 =>
        fromNative(native.index_select(dim.toLong, index.to(dtype = torch.int64).native))
  }

  // mask bool
  def masked_select[D1 <: DType](mask: Tensor[Bool]): Tensor[D] = fromNative(
    native.masked_select(mask.native)
  )

  def nonzero(un_used: Int*): Tensor[D] = fromNative(native.nonzero())

  def nonzero: Tensor[D] = fromNative(native.nonzero())

  def nonzero_static(size: Int, fill_value: Int = 1): Tensor[D] = fromNative(
    native.nonzero_static(size.toLong, fill_value.toLong)
  )

  def nonzero_static(size: Int): Tensor[D] = fromNative(native.nonzero_static(size.toLong))

  def nonzero_numpy(un_used: Int*): Seq[Tensor[D]] = {
    val res = native.nonzero_numpy()
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }

  def argwhere(un_used: Int*): Tensor[D] = fromNative(native.argwhere())

  def nonzero_numpy: Seq[Tensor[D]] = {
    val res = native.nonzero_numpy()
    var it = res.begin()
    val tensorList = new ListBuffer[Tensor[D]]()
    while (!it.equals(res.end())) {
      tensorList.append(fromNative(it.get()))
      it = it.increment()
    }
    tensorList.toSeq
  }
  def argwhere: Tensor[D] = fromNative(native.argwhere())

  def gather(
      dim: Long,
      index: Tensor[Int64] | Tensor[Int32],
      sparse_grad: Boolean = false
  ): Tensor[D] = {
    index.dtype match
      case torch.int64 => fromNative(native.gather(dim.toLong, index.native, sparse_grad))
      case torch.int32 =>
        fromNative(native.gather(dim.toLong, index.to(dtype = torch.int64).native, sparse_grad))
  }

  def gather(dim: Int, index: Tensor[Int64] | Tensor[Int32]): Tensor[D] = {
    index.dtype match
      case torch.int64 => fromNative(native.gather(dim.toLong, index.native))
      case torch.int32 =>
        fromNative(native.gather(dim.toLong, index.to(dtype = torch.int64).native))
  }

  def addcmul[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: Tensor[D1],
      value: S
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.addcmul(t1.native, t2.native, toScalar(value))
  )

  def addcdiv[D1 <: DType, S <: ScalaType](
      t1: Tensor[D1],
      t2: Tensor[D1],
      value: S
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.addcdiv(t1.native, t2.native, toScalar(value))
  )

  def addcmul_[D1 <: DType, S <: ScalaType](t1: Tensor[D1], t2: Tensor[D1], value: S): this.type = {
    native.addcmul_(t1.native, t2.native, toScalar(value))
    this
  }

  def addcdiv_[D1 <: DType, S <: ScalaType](t1: Tensor[D1], t2: Tensor[D1], value: S): this.type = {
    native.addcdiv_(t1.native, t2.native, toScalar(value))
    this
  }

  def addcmul[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.addcmul(t1.native, t2.native)
  )

  def addcdiv[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.addcdiv(t1.native, t2.native)
  )

  def addcmul_[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1]): this.type = {
    native.addcmul_(t1.native, t2.native)
    this
  }

  def addcdiv_[D1 <: DType](t1: Tensor[D1], t2: Tensor[D1]): this.type = {
    native.addcdiv_(t1.native, t2.native)
    this
  }

  def triangular_solve[D1 <: DType, D2 <: DType](
      a: Tensor[D1],
      upper: Boolean = true,
      transpose: Boolean = false,
      unitriangular: Boolean = false
  ): (Tensor[D2], Tensor[D2]) = {
    val res = native.triangular_solve(a.native, upper, transpose, unitriangular)
    val t = fromNative[D2](res.get0())
    val s = fromNative[D2](res.get1())
    (t, s)
  }
  def triangular_solve[D1 <: DType, D2 <: DType](a: Tensor[D1]): (Tensor[D2], Tensor[D2]) = {
    val res = native.triangular_solve(a.native)
    val t = fromNative[D2](res.get0())
    val s = fromNative[D2](res.get1())
    (t, s)
  }
  def svd[D1 <: DType](
      some: Boolean = true,
      compute_uv: Boolean = true
  ): (Tensor[D1], Tensor[D1], Tensor[D1]) = {
    val res = native.svd(some, compute_uv)
    val s = fromNative[D1](res.get0())
    val v = fromNative[D1](res.get1())
    val d = fromNative[D1](res.get2())
    (s, v, d)
  }
  def svd[D1 <: DType](): (Tensor[D1], Tensor[D1], Tensor[D1]) = {
    val res = native.svd()
    val s = fromNative[D1](res.get0())
    val v = fromNative[D1](res.get1())
    val d = fromNative[D1](res.get2())
    (s, v, d)
  }
  def swapaxes_(axis0: Long, axis1: Long): this.type = {
    native.swapaxes_(axis0, axis1)
    this
  }

  def swapdims_(dim0: Long, dim1: Long): this.type = {
    native.swapdims_(dim0, dim1)
    this
  }

  def swapaxes(axis0: Long, axis1: Long): Tensor[D] = fromNative(native.swapaxes(axis0, axis1))

  def swapdims(dim0: Long, dim1: Long): Tensor[D] = fromNative(native.swapdims(dim0, dim1))

  def cholesky(un_used: Int*): Tensor[D] = fromNative(native.cholesky())

  def cholesky: Tensor[D] = fromNative(native.cholesky())

  def cholesky(upper: Boolean = false): Tensor[D] = fromNative(native.cholesky(upper))

  def cholesky_solve[D1 <: DType](
      input2: Tensor[D1],
      upper: Boolean = false
  ): Tensor[Promoted[D1, D]] = fromNative(native.cholesky_solve(input2.native, upper))

  def cholesky_solve[D1 <: DType](input2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.cholesky_solve(input2.native)
  )
  def cholesky_inverse(upper: Boolean = false): Tensor[D] = fromNative(
    native.cholesky_inverse(upper)
  )
  def cholesky_inverse(): Tensor[D] = fromNative(native.cholesky_inverse())
  def geqrf[D1 <: DType, D2 <: DType](): (Tensor[D1], Tensor[D2]) = {
    val res = native.geqrf()
    val q = fromNative[D1](res.get0())
    val r = fromNative[D2](res.get1())
    (q, r)

  }
  def qr[D1 <: DType, D2 <: DType](some: Boolean = true): (Tensor[D1], Tensor[D2]) = {
    val res = native.qr(some)
    val q = fromNative[D1](res.get0())
    val r = fromNative[D2](res.get1())
    (q, r)

  }
  def qr[D1 <: DType, D2 <: DType](): (Tensor[D1], Tensor[D2]) = {
    val res = native.qr()
    val q = fromNative[D1](res.get0())
    val r = fromNative[D2](res.get1())
    (q, r)

  }
  def orgqr[D1 <: DType](input2: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.orgqr(input2.native)
  )
  def ormqr[D1 <: DType](
      input2: Tensor[D1],
      input3: Tensor[D1],
      left: Boolean = true,
      transpose: Boolean = true
  ): Tensor[Promoted[D1, D]] = fromNative(
    native.ormqr(input2.native, input3.native, left, transpose)
  )
  def ormqr[D1 <: DType](input2: Tensor[D1], input3: Tensor[D1]): Tensor[Promoted[D1, D]] =
    fromNative(native.ormqr(input2.native, input3.native))
  // todo need make sure
  def lu_solve[D1 <: DType, D2 <: DType](
      LU_data: Tensor[D1],
      LU_pivots: Tensor[D2]
  ): Tensor[Promoted[D1, D]] = fromNative(native.lu_solve(LU_data.native, LU_pivots.native))

  def multinomial(num_samples: Int): Tensor[D] = fromNative(native.multinomial(num_samples.toLong))
  def lgamma_(): this.type = {
    native.lgamma_()
    this
  }

  def lgamma(un_used: Int*): Tensor[D] = fromNative(native.lgamma())

  def digamma(un_used: Int*): Tensor[D] = fromNative(native.digamma())

  def lgamma: Tensor[D] = fromNative(native.lgamma())

  def digamma: Tensor[D] = fromNative(native.digamma())

  def polygamma_(n: Int): this.type = {
    native.polygamma_(n.toLong)
    this
  }

  def polygamma(n: Int): Tensor[D] = fromNative(native.polygamma(n.toLong))

  def erfinv_(): this.type = {
    native.erfinv_()
    this
  }

  def erfinv(un_used: Int*): Tensor[D] = fromNative(native.erfinv())

  def i0(un_used: Int*): Tensor[D] = fromNative(native.i0())

  def erfinv: Tensor[D] = fromNative(native.erfinv())

  def i0: Tensor[D] = fromNative(native.i0())

  def i0_(): this.type = {
    native.i0_()
    this
  }

  def sign_(): this.type = {
    native.sign_()
    this
  }

  def sign(un_used: Int*): Tensor[D] = fromNative(native.sign())

  def signbit(un_used: Int*): Tensor[D] = fromNative(native.signbit())

  def sign: Tensor[D] = fromNative(native.sign())

  def signbit: Tensor[D] = fromNative(native.signbit())

  def dist[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.dist(other.native)
  )

  def dist[D1 <: DType, S <: ScalaType](other: Tensor[D1], p: S): Tensor[Promoted[D1, D]] =
    fromNative(
      native.dist(other.native, toScalar(p))
    )

  def atan2_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.atan2_(other.native)
    this
  }

  def arctan2_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.arctan2_(other.native)
    this
  }

  def atan2[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.atan2(other.native)
  )

  def arctan2[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.arctan2(other.native)
  )
  def lerp[D1 <: DType](
      ends: Tensor[Int64] | Tensor[Int32],
      weight: Tensor[D1]
  ): Tensor[Promoted[D1, D]] = {
    ends.dtype match
      case torch.int64 => fromNative(native.lerp(ends.native, weight.native))
      case torch.int32 =>
        fromNative(native.lerp(ends.to(dtype = torch.int64).native, weight.native))
  }

  def lerp[D1 <: DType, S <: ScalaType](
      ends: Tensor[Int64] | Tensor[Int32],
      weight: S
  ): Tensor[Promoted[D1, D]] = {
    ends.dtype match
      case torch.int64 => fromNative(native.lerp(ends.native, toScalar(weight)))
      case torch.int32 =>
        fromNative(native.lerp(ends.to(dtype = torch.int64).native, toScalar(weight)))
  }

  def histc[D <: DType, S <: ScalaType](bins: Long, min: S, max: S): Tensor[D] = fromNative(
    native.histc(bins, toScalar(min), toScalar(max))
  )

  def histc[D <: DType](): Tensor[D] = fromNative(native.histc())

  def histogram[D1 <: DType, D2 <: DType](
      bins: Int,
      range: Seq[Double],
      weight: Option[Tensor[D1]],
      density: Boolean = false
  ): (Tensor[D1], Tensor[D2]) = {
    val his =
      native.histogram(bins.toLong, range.toArray, new TensorOptional(weight.get.native), density)
    val h1 = fromNative[D1](his.get0())
    val h2 = fromNative[D2](his.get1())
    (h1, h2)
  }

  def histogram[D1 <: DType, D2 <: DType](
      bins: Tensor[Int64] | Tensor[Int32]
  ): (Tensor[D1], Tensor[D2]) = {
    val his = bins.dtype match
      case torch.int64 => native.histogram(bins.native)
      case torch.int32 => native.histogram(bins.to(dtype = torch.int64).native)
    val h1 = fromNative[D1](his.get0())
    val h2 = fromNative[D2](his.get1())
    (h1, h2)
  }
  def histogram[D1 <: DType, D2 <: DType](): (Tensor[D1], Tensor[D2]) = {
    val his = native.histogram()
    val h1 = fromNative[D1](his.get0())
    val h2 = fromNative[D2](his.get1())
    (h1, h2)
  }

  def fmod_[D1 <: DType, S <: ScalaType](other: S): this.type = {
    native.fmod_(toScalar(other))
    this
  }

  def fmod[D1 <: DType, S <: ScalaType](other: S): Tensor[D1] = fromNative(
    native.fmod(toScalar(other))
  )

  def fmod_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.fmod_(other.native)
    this
  }

  def fmod[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.fmod(other.native)
  )
  def hypot[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.hypot(other.native)
  )
  def hypot_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.hypot_(other.native)
    this
  }

  def igamma[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.igamma(other.native)
  )

  def igamma_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.igamma_(other.native)
    this
  }

  def igammac_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.igammac_(other.native)
    this
  }

  def igammac[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.igammac(other.native)
  )

  def nextafter_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.nextafter_(other.native)
    this
  }

  def nextafter[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.nextafter(other.native)
  )

  def %=[D1 <: DType, S <: ScalaType](other: S): this.type = {
    native.remainder_(toScalar(other))
    this
  }

  def remainder_[D1 <: DType, S <: ScalaType](other: S): this.type = {
    native.remainder_(toScalar(other))
    this
  }

  def %[D1 <: DType, S <: ScalaType](other: S): Tensor[D1] = fromNative(
    native.remainder(toScalar(other))
  )
  def remainder[D1 <: DType, S <: ScalaType](other: S): Tensor[D1] = fromNative(
    native.remainder(toScalar(other))
  )

  def %=[D1 <: DType](other: Tensor[D1]): this.type = {
    native.remainder_(other.native)
    this
  }

  def remainder_[D1 <: DType](other: Tensor[D1]): this.type = {
    native.remainder_(other.native)
    this
  }

  def %[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.remainder(other.native)
  )
  def remainder[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.remainder(other.native)
  )

  def renorm_[S <: ScalaType](p: S, dim: Int, maxnorm: S): this.type = {
    native.renorm_(toScalar(p), dim.toLong, toScalar(maxnorm))
    this
  }
  def renorm_(p: Float, dim: Int, maxnorm: Float): this.type = {
    native.renorm_(toScalar(p), dim.toLong, toScalar(maxnorm))
    this
  }

  def pow_[S <: ScalaType](exponent: S): this.type = {
    native.pow_(toScalar(exponent))
    this
  }

  def pow_[D1 <: DType](exponent: Tensor[D1]): this.type = {
    native.pow_(exponent.native)
    this
  }

  def float_power_[S <: ScalaType](exponent: S): this.type = {
    native.float_power_(toScalar(exponent))
    this
  }
  def float_power_[D1 <: DType](exponent: Tensor[D1]): this.type = {
    native.float_power_(exponent.native)
    this
  }

  def normal_(): this.type = {
    native.normal_()
    this
  }

  def normal_(mean: Double, std: Double): this.type = {
    native.normal_(mean, std, new GeneratorOptional())
    this
  }

//  def min(): Tensor[D]= fromNative(native.min())

  def fmin[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.fmin(other.native)
  )

  def fmax[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.fmax(other.native)
  )

//  def maximum(other: Tensor[D]): Tensor[D]= fromNative(native.maximum(other.native))

  def max[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.max(other.native)
  )
//  def minimum(other: Tensor[D]): Tensor[D]= fromNative(native.minimum(other.native))
  def min[D1 <: DType](other: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.min(other.native)
  )

  def quantile[D1 <: DType](q: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.quantile(q.native)
  )
  def quantile[D1 <: DType](q: Double): Tensor[Promoted[D1, D]] = fromNative(native.quantile(q))
  def nanquantile[D1 <: DType](q: Double): Tensor[Promoted[D1, D]] = fromNative(
    native.nanquantile(q)
  )
  def nanquantile[D1 <: DType](q: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.nanquantile(q.native)
  )

  def msort(un_used: Int*): Tensor[D] = fromNative(native.msort())

  def sort(un_used: Int*): (Tensor[D], Tensor[D]) = {
    val res = native.sort()
    val s = fromNative[D](res.get0())
    val t = fromNative[D](res.get1())
    (s, t)
  }

  def msort: Tensor[D] = fromNative(native.msort())

  def sort: (Tensor[D], Tensor[D]) = {
    val res = native.sort()
    val s = fromNative[D](res.get0())
    val t = fromNative[D](res.get1())
    (s, t)
  }
  def argsort(stable: Boolean): Tensor[D] = fromNative(native.argsort(stable))

  def argsort(un_used: Int*): Tensor[D] = fromNative(native.argsort())

  def argsort: Tensor[D] = fromNative(native.argsort())

  def topk(
      k: Int,
      dim: Int,
      largest: Boolean = true,
      sorted: Boolean = true
  ): (Tensor[D], Tensor[Int64]) = {

    val res = native.topk(k.toLong, dim.toLong, largest, sorted)
    val s = fromNative[D](res.get0())
    val t = fromNative[Int64](res.get1())
    (s, t)
  }
  def topk(k: Int): (Tensor[D], Tensor[Int64]) = {
    val res = native.topk(k.toLong)
    val s = fromNative[D](res.get0())
    val t = fromNative[Int64](res.get1())
    (s, t)
  }

  def unfold(dimension: Int, size: Int, step: Int): Tensor[D] = fromNative(
    native.unfold(dimension.toLong, size.toLong, step.toLong)
  )
  def float_power[D1 <: DType](exponent: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.float_power(exponent.native)
  )

  def float_power[D1 <: DType, S <: ScalaType](exponent: S): Tensor[D] = fromNative(
    native.float_power(toScalar(exponent))
  )

  def pow[D1 <: DType](exponent: Tensor[D1]): Tensor[Promoted[D1, D]] = fromNative(
    native.pow(exponent.native)
  )

  def renorm(p: Float, dim: Int, maxnorm: Float): Tensor[D] = fromNative(
    native.renorm(toScalar(p), dim.toLong, toScalar(maxnorm))
  )

  def renorm[S <: ScalaType](p: S, dim: Int, maxnorm: S): Tensor[Promoted[D, ScalaToDType[S]]] =
    fromNative(
      native.renorm(toScalar(p), dim.toLong, toScalar(maxnorm))
    )

  def alias(un_used: Int*): Tensor[D] = fromNative(native.alias())

  def isfinite(un_used: Int*): Tensor[D] = fromNative(native.isfinite())

  def isinf(un_used: Int*): Tensor[D] = fromNative(native.isinf())

  def isposinf(un_used: Int*): Tensor[D] = fromNative(native.isposinf())

  def isneginf(un_used: Int*): Tensor[D] = fromNative(native.isneginf())

  def det(un_used: Int*): Tensor[D] = fromNative(native.det())

  def alias: Tensor[D] = fromNative(native.alias())

  def isfinite: Tensor[D] = fromNative(native.isfinite())

  def isinf: Tensor[D] = fromNative(native.isinf())

  def isposinf: Tensor[D] = fromNative(native.isposinf())

  def isneginf: Tensor[D] = fromNative(native.isneginf())

  def det: Tensor[D] = fromNative(native.det())

  def slogdet[D1 <: DType](): (Tensor[D], Tensor[D1]) = {
    val detTuple = native.slogdet()
    val d1 = fromNative[D](detTuple.get0())
    val d2 = fromNative[D1](detTuple.get1())
    (d1, d2)
  }

  def logdet(un_used: Int*): Tensor[D] = fromNative(native.logdet())

  def inverse(un_used: Int*): Tensor[D] = fromNative(native.inverse())

  def logdet: Tensor[D] = fromNative(native.logdet())

  def inverse: Tensor[D] = fromNative(native.inverse())

  def inner(other: Tensor[D]): Tensor[D] = fromNative(native.inner(other.native))

  def outer(vec2: Tensor[D]): Tensor[D] = fromNative(native.outer(vec2.native))

  def ger(vec2: Tensor[D]): Tensor[D] = fromNative(native.ger(vec2.native))

  def to_padded_tensor(padding: Double): Tensor[D] = fromNative(native.to_padded_tensor(padding))

  def `var`(dim: Int): Tensor[D] = fromNative(native.`var`(dim))

  def std(dim: Int): Tensor[D] = fromNative(native.std(dim))
//  def requires_grad_(): Tensor[D] = fromNative(native.requires_grad_())
//
//  def requires_grad_(requires_grad: Boolean= true): Tensor[D] = fromNative(native.requires_grad_(requires_grad))

  /** Returns the sum of the elements of the diagonal of the input 2-D matrix. */
  def trace: Tensor[D] = fromNative(native.trace)

  /** Returns a summary of the contents of this tensor.
    *
    * @param maxEntries
    *   Maximum number of entries to show for each axis/dimension. If the size of an axis exceeds
    *   `maxEntries`, the output of that axis will be shortened to the first and last three
    *   elements. Defaults to `6`. Values below `6` are ignored.
    * @param flattened
    *   If `true`, the summary is flattened to one line. Otherwise, the summary may span multiple
    *   lines.
    * @param includeInfo
    *   If `true`, the data type and the shape of the tensor are explicitly included in the summary.
    *   Otherwise, they are not.
    * @return
    *   Tensor summary.
    */
  def summarize(
      maxEntries: Int = 6,
      flattened: Boolean = false,
      includeInfo: Boolean = true
  ): String =
    def format(x: Any): String =
      x match
        case x: Float  => "%1.4f".format(x)
        case x: Double => "%1.4f".format(x)
        case x         => x.toString

    def summarize(tensor: Tensor[D], maxEntries: Int): String =
      tensor.dim match
        case 0 => format(tensor.toSeq.head)
        case 1 =>
          val slice =
            if tensor.numel <= math.max(maxEntries, 6) then tensor.toSeq.map(format)
            else
              val left = tensor(Slice(0, maxEntries / 2)).toSeq.map(format)
              val right = tensor(Slice(-maxEntries / 2)).toSeq.map(format)
              left ++ Seq("...") ++ right
          slice.mkString("[", ", ", "]")
        case _ =>
          val innerSummary = {
            def summarizeSlice(index: Int) = summarize(tensor(index), maxEntries)
            val sliceLen = tensor.size(0)
            if sliceLen <= math.max(maxEntries, 6) then
              for (i <- 0 until sliceLen) yield summarizeSlice(i)
            else
              val start = for (i <- 0 until maxEntries / 2) yield summarizeSlice(i)
              val end = for (i <- sliceLen - maxEntries / 2 until sliceLen) yield summarizeSlice(i)
              (start :+ "...") ++ end
          }
          val padding = " " * (this.dim - tensor.dim + 1)
          val extraLine = if (!flattened && tensor.dim >= 3) "\n" else ""
          innerSummary.mkString("[", (if (!flattened) ",\n" else ", ") + extraLine + padding, "]")

    if dtype == undefined then "undefined tensor"
    else if includeInfo then
      info + " " + (if !flattened then "\n" else ": ") + summarize(this, maxEntries)
    else summarize(this, maxEntries)

  def view(shape: Int*): Tensor[D] = fromNative(native.view(shape.map(_.toLong)*))

  def info: String =
    s"tensor dtype=${dtype.toString}, shape=${size.mkString("[", ", ", "]")}, device=${device.device}"

  override def toString: String = summarize()

  private[torch] def requireNativeType(expected: ScalarType) = require(
    native.scalar_type().intern() == expected,
    s"Expected native tensor type $expected, got ${native.scalar_type().intern()}"
  )
}

sealed class UInt8Tensor(native: pytorch.Tensor) extends Tensor[UInt8](native) { /* 0, Byte */
  require(native.scalar_type().intern() == ScalarType.Byte)
  override def dtype: UInt8 = uint8
}

sealed class Int8Tensor(native: pytorch.Tensor) extends Tensor[Int8](native) { /* 1, Char */
  requireNativeType(ScalarType.Char)
  override def dtype: Int8 = int8
}
sealed class Int16Tensor(native: pytorch.Tensor) extends Tensor[Int16](native) { /* 2, Short */
  requireNativeType(ScalarType.Short)
  override def dtype: Int16 = int16
}
sealed class Int32Tensor(native: pytorch.Tensor) extends Tensor[Int32](native) { /* 3, Int */
  requireNativeType(ScalarType.Int)
  override def dtype: Int32 = int32
}
sealed class Int64Tensor(native: pytorch.Tensor) extends Tensor[Int64](native) { /* 4, Long */
  requireNativeType(ScalarType.Long)
  override def dtype: Int64 = int64
}
sealed class Float16Tensor(native: pytorch.Tensor) extends Tensor[Float16](native) { /* 5, Half */
  requireNativeType(ScalarType.Half)
  override def dtype: Float16 = float16
}
sealed class Float32Tensor(native: pytorch.Tensor) extends Tensor[Float32](native) { /* 6, Float */
  requireNativeType(ScalarType.Float)
  override def dtype: Float32 = float32
}
sealed class Float64Tensor(native: pytorch.Tensor) extends Tensor[Float64](native) { /* 7, Double */
  requireNativeType(ScalarType.Double)
  override def dtype: Float64 = float64
}
sealed class Complex32Tensor(native: pytorch.Tensor) extends Tensor[Complex32](native) { /* 8, ComplexHalf */
  requireNativeType(ScalarType.ComplexHalf)
  override def dtype: Complex32 = complex32
}
sealed class Complex64Tensor(native: pytorch.Tensor) extends Tensor[Complex64](native) { /* 9, ComplexFloat */
  requireNativeType(ScalarType.ComplexFloat)
  override def dtype: Complex64 = complex64
}
sealed class Complex128Tensor(native: pytorch.Tensor) extends Tensor[Complex128](native) { /* 10, ComplexDouble */
  requireNativeType(ScalarType.ComplexDouble)
  override def dtype: Complex128 = complex128
}
sealed class BoolTensor(native: pytorch.Tensor) extends Tensor[Bool](native) { /* 11 Bool */
  requireNativeType(ScalarType.Bool)
  override def dtype: Bool = bool
}
sealed class QInt8Tensor(native: pytorch.Tensor) extends Tensor[QInt8](native) { /* 12 */
  requireNativeType(ScalarType.QInt8)
  override def dtype: QInt8 = qint8
}
sealed class QUInt8Tensor(native: pytorch.Tensor) extends Tensor[QUInt8](native) { /* 13 */
  requireNativeType(ScalarType.QUInt8)
  override def dtype: QUInt8 = quint8
}
sealed class QInt32Tensor(native: pytorch.Tensor) extends Tensor[QInt32](native) { /* 14 */
  requireNativeType(ScalarType.QInt32)
  override def dtype: QInt32 = qint32
}
sealed class BFloat16Tensor(native: pytorch.Tensor) extends Tensor[BFloat16](native) { /* 15 */
  requireNativeType(ScalarType.BFloat16)
  override def dtype: BFloat16 = bfloat16
}
sealed class QUInt4x2Tensor(native: pytorch.Tensor) extends Tensor[QUInt4x2](native) { /* 16 */
  requireNativeType(ScalarType.QUInt4x2)
  override def dtype: QUInt4x2 = quint4x2
}
sealed class QUInt2x4Tensor(native: pytorch.Tensor) extends Tensor[QUInt2x4](native) { /* 16 */
  requireNativeType(ScalarType.QUInt2x4)
  override def dtype: QUInt2x4 = quint2x4
}
sealed class Bits1x8Tensor(native: pytorch.Tensor) extends Tensor[Bits1x8](native) { /* 18 */
  requireNativeType(ScalarType.Bits1x8)
  override def dtype: Bits1x8 = bits1x8
}
sealed class Bits2x4Tensor(native: pytorch.Tensor) extends Tensor[Bits2x4](native) { /* 19 */
  requireNativeType(ScalarType.Bits2x4)
  override def dtype: Bits2x4 = bits2x4
}
sealed class Bits4x2Tensor(native: pytorch.Tensor) extends Tensor[Bits4x2](native) { /* 20 */
  requireNativeType(ScalarType.Bits4x2)
  override def dtype: Bits4x2 = bits4x2
}
sealed class Bits8Tensor(native: pytorch.Tensor) extends Tensor[Bits8](native) { /* 21 */
  requireNativeType(ScalarType.Bits8)
  override def dtype: Bits8 = bits8
}
sealed class Bits16Tensor(native: pytorch.Tensor) extends Tensor[Bits16](native) { /* 22 */
  requireNativeType(ScalarType.Bits16)
  override def dtype: Bits16 = bits16
}
sealed class Float8_e5m2Tensor(native: pytorch.Tensor) extends Tensor[Float8_e5m2](native) { /* 23 */
  requireNativeType(ScalarType.Float8_e5m2)
  override def dtype: Float8_e5m2 = float8_e5m2
}
sealed class Float8_e4m3fnTensor(native: pytorch.Tensor) extends Tensor[Float8_e4m3fn](native) { /* 34 */
  requireNativeType(ScalarType.Float8_e4m3fn)
  override def dtype: Float8_e4m3fn = float8_e4m3fn
}
sealed class UndefinedTensor(native: pytorch.Tensor) extends Tensor[Undefined](native) { /* 25 */
  requireNativeType(ScalarType.Undefined)
  override def dtype: Undefined = undefined
}
sealed class NumOptionsTensor(native: pytorch.Tensor) extends Tensor[NumOptions](native) { /* 26 */
  requireNativeType(ScalarType.NumOptions)
  override def dtype: NumOptions = numoptions
}

type IntTensor = UInt8Tensor | Int8Tensor | Int16Tensor | Int32Tensor | Int64Tensor
type ComplexTensor = Complex32Tensor | Complex64Tensor | Complex128Tensor

object Tensor:

  def fromNative[D <: DType](native: pytorch.Tensor): Tensor[D] =
    (native.scalar_type().intern() match
      case ScalarType.Byte          => new UInt8Tensor(native)
      case ScalarType.Char          => new Int8Tensor(native)
      case ScalarType.Short         => new Int16Tensor(native)
      case ScalarType.Int           => new Int32Tensor(native)
      case ScalarType.Long          => new Int64Tensor(native)
      case ScalarType.Half          => new Float16Tensor(native)
      case ScalarType.Float         => new Float32Tensor(native)
      case ScalarType.Double        => new Float64Tensor(native)
      case ScalarType.ComplexHalf   => new Complex32Tensor(native)
      case ScalarType.ComplexFloat  => new Complex64Tensor(native)
      case ScalarType.ComplexDouble => new Complex128Tensor(native)
      case ScalarType.Bool          => new BoolTensor(native)
      case ScalarType.QInt8         => new QInt8Tensor(native)
      case ScalarType.QUInt8        => new QUInt8Tensor(native)
      case ScalarType.QInt32        => new QInt32Tensor(native)
      case ScalarType.BFloat16      => new BFloat16Tensor(native)
      case ScalarType.QUInt4x2      => new QUInt4x2Tensor(native)
      case ScalarType.QUInt2x4      => new QUInt2x4Tensor(native)
      case ScalarType.Bits1x8       => new Bits1x8Tensor(native)
      case ScalarType.Bits2x4       => new Bits2x4Tensor(native)
      case ScalarType.Bits4x2       => new Bits4x2Tensor(native)
      case ScalarType.Bits8         => new Bits8Tensor(native)
      case ScalarType.Bits16        => new Bits16Tensor(native)
      case ScalarType.Float8_e5m2   => new Float8_e5m2Tensor(native)
      case ScalarType.Float8_e4m3fn => new Float8_e4m3fnTensor(native)
      case ScalarType.Undefined     => new UndefinedTensor(native)
      case ScalarType.NumOptions    => new NumOptionsTensor(native)
    ).asInstanceOf[Tensor[D]]

  def apply[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean,
      device: Device
  ): Tensor[ScalaToDType[U]] = {
    val tensor = this.apply(data, Strided, device, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

//  def apply[U <: ScalaType : ClassTag](
//                                        NdArray: NDArray[U],
//                                        requires_grads: Boolean = false,
//                                        devices: Device = CPU
//                                      ): Tensor[ScalaToDType[U]] =
//    Tensor.createFromNDArray(data = NdArray, requires_grad = requires_grads, device = devices)

  def createFromNDArray[U <: ScalaType: ClassTag](
      data: NDArray[U],
      requires_grad: Boolean,
      device: Device
  ): Tensor[ScalaToDType[U]] = {
    require(data.getNdim <= 5, "Only 1D, 2D, and 3D, 4D, 5D arrays are supported")
    val shapeSize = data.getShape.size
    val ndArray = data.getArray
    val tensor: Tensor[ScalaToDType[U]] = (data.getArray, shapeSize) match {
      case (singleDim: Array[U], 0) =>
        val dataSeq = singleDim.toSeq.asInstanceOf[Seq[U]]
        Tensor(dataSeq, Strided, device, requires_grad)
      case (singleDim: Array[U], 1) =>
        val dataSeq = singleDim.toSeq.asInstanceOf[Seq[U]]
        Tensor(dataSeq, Strided, device, requires_grad)
      case (twoDim: Array[Array[U]], 2) =>
        val dataSeq = twoDim.map((arr: Array[U]) => arr.toSeq).toSeq
        this.apply(dataSeq, Strided, device, requires_grad)
      case (threeDim: Array[Array[Array[U]]], 3) =>
        val dataSeq = threeDim.map((arr: Array[Array[U]]) => arr.map(_.toSeq).toSeq).toSeq
        this.apply(dataSeq, Strided, device, requires_grad)
      case (fourDim: Array[Array[Array[Array[U]]]], 4) =>
        val dataSeq =
          fourDim.map((arr: Array[Array[Array[U]]]) => arr.map(_.map(_.toSeq).toSeq).toSeq).toSeq
        this.apply(dataSeq, Strided, device, requires_grad)
      case (fiveDim: Array[Array[Array[Array[Array[U]]]]], 5) =>
        val dataSeq = fiveDim
          .map((arr: Array[Array[Array[Array[U]]]]) =>
            arr.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq
          )
          .toSeq
        this.apply(dataSeq, Strided, device, requires_grad)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported array dimension： shapeSize $shapeSize")
    }
    tensor
  }

  //    val tensor2 : Tensor[ScalaToDType[U]] =  ndArray match {
  //      case singleDim: Array[U] =>
  //        val dataSeq = singleDim.toSeq
  //        this.apply(dataSeq, Strided, device, requires_grad)
  //      case twoDim: Array[Array[U]] =>
  //        val dataSeq = twoDim.map((arr:Array[U]) => arr.toSeq).toSeq
  //        this.apply(dataSeq, Strided, device, requires_grad)
  //      case threeDim: Array[Array[Array[U]]] =>
  //        val dataSeq = threeDim.map((arr:Array[Array[U]]) => arr.map(_.toSeq).toSeq).toSeq
  //        this.apply(dataSeq, Strided, device, requires_grad)
  //      case fourDim: Array[Array[Array[Array[U]]]] =>
  //        val dataSeq = fourDim.map((arr: Array[Array[Array[U]]]) => arr.map(_.map(_.toSeq).toSeq).toSeq).toSeq
  //        Tensor(dataSeq, Strided, device, requires_grad)
  //      case fiveDim: Array[Array[Array[Array[Array[U]]]]] =>
  //        val dataSeq = fiveDim.map((arr: Array[Array[Array[Array[U]]]]) => arr.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq).toSeq
  //        Tensor(dataSeq, Strided, device, requires_grad)
  //      case _ => throw new IllegalArgumentException("Unsupported array dimension")
  //    }
  //    tensor

  def arrayToSeq[U <: ScalaType: ClassTag](
      arr: Array[U] | Array[Array[U]] | Array[Array[Array[U]]] | Array[Array[Array[Array[U]]]] |
        Array[Array[Array[Array[Array[U]]]]]
  ): U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
    Seq[Seq[Seq[Seq[Seq[U]]]]] = {
    arr match {
      case singleDim: Array[U]              => singleDim.toSeq
      case twoDim: Array[Array[U]]          => twoDim.map(_.toSeq).toSeq
      case threeDim: Array[Array[Array[U]]] => threeDim.map(_.map(_.toSeq).toSeq).toSeq
      case fourDim: Array[Array[Array[Array[U]]]] =>
        val seq = fourDim.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq
        seq
      case fiveDim: Array[Array[Array[Array[Array[U]]]]] =>
        val seq = fiveDim.map(_.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq).toSeq
        seq
//      case sixDim: Array[Array[Array[Array[Array[Array[U]]]]]] =>
//        val seq = sixDim.map(_.map(_.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq).toSeq).toSeq
//        seq
      case _ => throw new IllegalArgumentException("Unsupported array dimension")
    }
  }

  def apply[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      requires_grad: Boolean
  ): Tensor[ScalaToDType[U]] = {
    val tensor = this.apply(data, Strided, CPU, requires_grad)
    tensor.requires_grad_(requires_grad)
    tensor
  }

  /** Constructs a tensor with no autograd history (also known as a “leaf tensor”) by copying data.
    */
  // TODO support arbitrary multidimensional arrays as input
  // TODO support explicit dtype
  def apply[U <: ScalaType: ClassTag](
      data: U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] | Seq[Seq[Seq[Seq[U]]]] |
        Seq[Seq[Seq[Seq[Seq[U]]]]] | NDArray[U],
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[ScalaToDType[U]] =
    data match
      case ndArray: NDArray[U] =>
        Tensor.createFromNDArray(data = ndArray, requires_grad = requiresGrad, device = device)
      case quintSeq(data) =>
        val tensor = apply(
          data.flatten.flatten.flatten.flatten.asInstanceOf[Seq[U]],
          layout,
          device,
          requiresGrad
        )
          .view(
            data.length,
            data.head.length,
            data.head.head.length,
            data.head.head.head.length,
            data.head.head.head.head.length
          )
//        tensor.set_requires_grad(requiresGrad)
        tensor.requires_grad_(requiresGrad)
        tensor
      case quadSeq(data) =>
        val tensor =
          apply(data.flatten.flatten.flatten.asInstanceOf[Seq[U]], layout, device, requiresGrad)
            .view(data.length, data.head.length, data.head.head.length, data.head.head.head.length)
//        tensor.set_requires_grad(requiresGrad)
        tensor.requires_grad_(requiresGrad)
        tensor
      case tripleSeq(data) =>
        val tensor = apply(data.flatten.flatten.asInstanceOf[Seq[U]], layout, device, requiresGrad)
          .view(data.length, data.head.length, data.head.head.length)
//        tensor.set_requires_grad(requiresGrad)
        tensor.requires_grad_(requiresGrad)
        tensor
      case doubleSeq(data) =>
        val tensor = apply(data.flatten.asInstanceOf[Seq[U]], layout, device, requiresGrad)
          .view(data.length, data.head.length)
//        tensor.set_requires_grad(requiresGrad)
        tensor.requires_grad_(requiresGrad)
        tensor
      case singleSeq(data) =>
        val (pointer, inputDType) =
          data.asInstanceOf[Seq[U]].toArray match
            case bools: Array[Boolean] =>
              (
                {
                  val p = new BoolPointer(bools.length)
                  for ((b, i) <- bools.zipWithIndex) p.put(i, b)
                  p
                },
                bool
              )
            case bytes: Array[Byte]     => (new BytePointer(ByteBuffer.wrap(bytes)), int8)
            case shorts: Array[Short]   => (new ShortPointer(ShortBuffer.wrap(shorts)), int16)
            case ints: Array[Int]       => (new IntPointer(IntBuffer.wrap(ints)), int32)
            case longs: Array[Long]     => (new LongPointer(LongBuffer.wrap(longs)), int64)
            case floats: Array[Float]   => (new FloatPointer(FloatBuffer.wrap(floats)), float32)
            case doubles: Array[Double] => (new DoublePointer(DoubleBuffer.wrap(doubles)), float64)
            case complexFloatArray(complexFloats) =>
              (
                new FloatPointer(
                  FloatBuffer.wrap(complexFloats.flatMap(c => Array(c.real, c.imag)))
                ),
                complex64
              )
            case complexDoubleArray(complexDoubles) =>
              (
                new DoublePointer(
                  DoubleBuffer.wrap(complexDoubles.flatMap(c => Array(c.real, c.imag)))
                ),
                complex128
              )
            case _ =>
              throw new IllegalArgumentException(
                s"Unsupported data type ${summon[ClassTag[U]].runtimeClass.getSimpleName}"
              )

        fromNative(
          torchNative
            .from_blob(
              pointer,
              Array(data.length.toLong),
              NativeConverters.tensorOptions(inputDType, layout, CPU, requiresGrad)
            )
            .clone()
        ).to(device = device)
//        tensor.set_requires_grad(requiresGrad)
//        tensor
      case data: U =>
        val tensor = fromScalar(data)
//        tensor.set_requires_grad(requiresGrad)
        tensor.requires_grad_(requiresGrad)
        tensor.to(device = device)
      case _ => throw new IllegalArgumentException("Unsupported type")

  def scalar_tensor[S <: ScalaType](
      s: S,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[ScalaToDType[S]] =
    val dtype = scalaToDType(s)
    fromNative(
      torchNative.scalar_tensor(
        NativeConverters.toScalar(s),
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

  def fromScalar[S <: ScalaType](
      s: S,
      layout: Layout = Strided,
      device: Device = CPU,
      requiresGrad: Boolean = false
  ): Tensor[ScalaToDType[S]] =
    val dtype = scalaToDType(s)
    fromNative(
      torchNative.scalar_tensor(
        NativeConverters.toScalar(s),
        NativeConverters.tensorOptions(dtype, layout, device, requiresGrad)
      )
    )

/** Scalar/Tensor extensions to allow tensor operations directly on scalars */
extension [S <: ScalaType](s: S)
  def +[D <: DType](t: Tensor[D]): Tensor[Promoted[D, ScalaToDType[S]]] = t.add(s)
  def -[D <: DType](t: Tensor[D]): Tensor[Promoted[D, ScalaToDType[S]]] = t.sub(s)
  def *[D <: DType](t: Tensor[D]): Tensor[Promoted[D, ScalaToDType[S]]] = t.mul(s)
  def /[D <: DType](t: Tensor[D]): Tensor[Div[D, ScalaToDType[S]]] = t.div(s)
  def **[D <: DType](t: Tensor[D])(using
      @implicitNotFound(""""pow" not implemented for bool""")
      ev1: Promoted[D, ScalaToDType[S]] NotEqual Bool,
      @implicitNotFound(""""pow" not implemented for complex32""")
      ev2: Promoted[D, ScalaToDType[S]] NotEqual Complex32
  ): Tensor[Promoted[D, ScalaToDType[S]]] = t.pow(s)
  def **[S2 <: ScalaType](other: S2): DTypeToScala[Promoted[ScalaToDType[S], ScalaToDType[S2]]] =
    Tensor.fromScalar(s).pow(other).item

//  def new_tensor(data: Seq[Long]): Tensor[torch.Int64] = {
//    torch.Tensor(data)
//  }

//  fromNative(
//    native.new_tensor(data)
//  )
