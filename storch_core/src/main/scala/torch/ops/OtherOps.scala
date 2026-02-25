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

import internal.NativeConverters.*
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{
  BoolOptional,
  DeviceOptional,
  DoubleVector,
  DoubleVectorOptional,
  FunctionSchema,
  GridSampleFuncOptions,
  GridSampleMode,
  GridSamplePaddingMode,
  InterpolateFuncOptions,
  InterpolateMode,
  LongArrayRef,
  LongArrayRefOptional,
  LongOptional,
  LongVector,
  LongVectorOptional,
  ProcessGroup,
  ScalarTypeOptional,
  Storage,
  Stream,
  StringViewOptional,
  SymInt,
  SymIntOptional,
  TensorOptional,
  TensorOptions,
  TensorVector,
  TypeMeta,
  VariableHooksInterface,
  WarningHandler,
  Work,
  kArea,
  kBicubic,
  kBilinear,
  kBorder,
  kLinear,
  kNearest,
  kNearestExact,
  kReflection,
  kTrilinear,
  kZeros
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}
import Device.CPU
import torch.Layout.{Sparse, SparseBsc, SparseBsr, SparseCsc, SparseCsr, Strided}
import torch.internal.NativeConverters
import torch.numpy.matrix.NDArray

import scala.reflect.ClassTag

/** Other Ops
  *
  * https://pytorch.org/docs/stable/torch.html#other-operations
  */
private[torch] trait OtherOps {

  /** Sums the product of the elements of the input `operands` along dimensions specified using a
    * notation based on the Einstein summation convention.
    *
    * Einsum allows computing many common multi-dimensional linear algebraic array operations by
    * representing them in a short-hand format based on the Einstein summation convention, given by
    * `equation`. The details of this format are described below, but the general idea is to label
    * every dimension of the input `operands` with some subscript and define which subscripts are
    * part of the output. The output is then computed by summing the product of the elements of the
    * `operands` along the dimensions whose subscripts are not part of the output. For example,
    * matrix multiplication can be computed using einsum as [torch.einsum(\"ij,jk-\>ik\", A, B)].
    * Here, j is the summation subscript and i and k the output subscripts (see section below for
    * more details on why).
    *
    * Equation:
    *
    * The `equation` string specifies the subscripts (letters in [\[a-zA-Z\]]) for each dimension of
    * the input `operands` in the same order as the dimensions, separating subscripts for each
    * operand by a comma (\',\'), e.g. [\'ij,jk\'] specify subscripts for two 2D operands. The
    * dimensions labeled with the same subscript must be broadcastable, that is, their size must
    * either match or be [1]. The exception is if a subscript is repeated for the same input
    * operand, in which case the dimensions labeled with this subscript for this operand must match
    * in size and the operand will be replaced by its diagonal along these dimensions. The
    * subscripts that appear exactly once in the `equation` will be part of the output, sorted in
    * increasing alphabetical order. The output is computed by multiplying the input `operands`
    * element-wise, with their dimensions aligned based on the subscripts, and then summing out the
    * dimensions whose subscripts are not part of the output.
    *
    * Optionally, the output subscripts can be explicitly defined by adding an arrow (\'-\>\') at
    * the end of the equation followed by the subscripts for the output. For instance, the following
    * equation computes the transpose of a matrix multiplication: \'ij,jk-\>ki\'. The output
    * subscripts must appear at least once for some input operand and at most once for the output.
    *
    * Ellipsis (\'\...\') can be used in place of subscripts to broadcast the dimensions covered by
    * the ellipsis. Each input operand may contain at most one ellipsis which will cover the
    * dimensions not covered by subscripts, e.g. for an input operand with 5 dimensions, the
    * ellipsis in the equation [\'ab\...c\'] cover the third and fourth dimensions. The ellipsis
    * does not need to cover the same number of dimensions across the `operands` but the \'shape\'
    * of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the
    * output is not explicitly defined with the arrow (\'-\>\') notation, the ellipsis will come
    * first in the output (left-most dimensions), before the subscript labels that appear exactly
    * once for the input operands. e.g. the following equation implements batch matrix
    * multiplication [\'\...ij,\...jk\'].
    *
    * A few final notes: the equation may contain whitespaces between the different elements
    * (subscripts, ellipsis, arrow and comma) but something like [\'. . .\'] is not valid. An empty
    * string [\'\'] is valid for scalar operands.
    *
    * Note: Sublist format it not supported yet
    *
    * Example:
    * ```scala sc
    * import torch.*
    * // trace
    * torch.einsum("ii", torch.randn(Seq(4, 4)))
    *
    * // diagonal
    * torch.einsum("ii->i", torch.randn(Seq(4, 4)))
    *
    * // outer product
    * val x = torch.randn(Seq(5))
    * val y = torch.randn(Seq(4))
    * torch.einsum("i,j->ij", x, y)
    *
    * // batch matrix multiplication
    * val As = torch.randn(Seq(3, 2, 5))
    * val Bs = torch.randn(Seq(3, 5, 4))
    * torch.einsum("bij,bjk->bik", As, Bs)
    *
    * // with sublist format and ellipsis
    * // Not supported yet in Storch
    * // torch.einsum(As, Seq(---, 0, 1), Bs, Seq(---, 1, 2), Seq(---, 0, 2))
    *
    * // batch permute
    * val A = torch.randn(Seq(2, 3, 4, 5))
    * torch.einsum("...ij->...ji", A).shape
    * ```
    *
    * ```scala sc
    * // equivalent to torch.nn.functional.bilinear
    * val A = torch.randn(Seq(3, 5, 4))
    * val l = torch.randn(Seq(2, 5))
    * val r = torch.randn(Seq(2, 4))
    * torch.einsum("bn,anm,bm->ba", l, A, r)
    * ```
    *
    * @group other_ops
    * @param equation
    *   The subscripts for the Einstein summation.
    * @param operands
    *   The tensors to compute the Einstein summation of.
    */
  def einsum[D <: DType](equation: String, operands: Tensor[D]*): Tensor[D] =
    // TODO the equation input is not yet working, see https://github.com/bytedeco/javacpp-presets/discussions/1390
    fromNative(torchNative.einsum(BytePointer(equation), toArrayRef(operands)))

  /** Returns the sum of the elements of the diagonal of the input 2-D matrix. */
  def trace[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(torchNative.trace(input.native))

  def from_native[D <: DType](rawTensor: org.bytedeco.pytorch.Tensor): Tensor[D] = fromNative(
    rawTensor
  )

  def requires_grad(flag: Boolean) = torchNative.requires_grad(flag)

  def get_default_dtype = torchNative.get_default_dtype()

  def get_default_complex_dtype = torchNative.get_default_complex_dtype()

  def set_default_dtype(dtype: TypeMeta) = torchNative.set_default_dtype(dtype)

  def block_diag[D <: DType](tensorList: Seq[Tensor[D]]): Tensor[D] =
    fromNative(torchNative.block_diag(new TensorVector(tensorList.map(_.native)*)))

  def isclose[D <: DType](tensor: Tensor[D], tensor2: Tensor[D]): Tensor[D] =
    fromNative(torchNative.isclose(tensor.native, tensor2.native))

  def linear[D <: DType](
      input: Tensor[D],
      weight: Tensor[D],
      bias: Option[Tensor[D]] = None
  ): Tensor[D] =
    if bias.isDefined then
      fromNative(
        torchNative.linear(input.native, weight.native, new TensorOptional(bias.get.native))
      )
    else fromNative(torchNative.linear(input.native, weight.native))

  def can_cast(dtype: DType, dtype2: DType): Boolean =
    torchNative.can_cast(dtype.toScalarType, dtype2.toScalarType)

  def from_numpy[U <: ScalaType: ClassTag](
      data: NDArray[U],
      requires_grad: Boolean = true
  ): Tensor[ScalaToDType[U]] = Tensor.createFromNDArray[U](data, requires_grad, CPU)

  def is_floating_point[D <: DType](tensor: Tensor[D]): Boolean =
    torchNative.is_floating_point(tensor.native)

  def is_distributed[D <: DType](tensor: Tensor[D]): Boolean =
    torchNative.is_distributed(tensor.native)

  def is_signed[D <: DType](tensor: Tensor[D]): Boolean = torchNative.is_signed(tensor.native)

  def is_inference[D <: DType](tensor: Tensor[D]): Boolean = torchNative.is_inference(tensor.native)

  def is_vulkan_available(): Boolean = torchNative.is_vulkan_available()

  def interpolate[D <: DType](
      x: Tensor[D],
      output_size: List[Long] = List(),
      scale_factor: List[Double],
      mode: String = "nearest",
      align_corners: Boolean = false,
      recompute_scale_factor: Boolean = false,
      antialias: Boolean = false
  ): Tensor[D] = {
    val options = new InterpolateFuncOptions
    val nativeMode = mode match {
      case "nearest" | "Nearest"                                                => new kNearest
      case "bilinear" | "Bilinear"                                              => new kBilinear
      case "bicubic" | "Bicubic"                                                => new kBicubic
      case "trilinear" | "Trilinear"                                            => new kTrilinear
      case "area" | "Area" | "kArea" | "KArea"                                  => new kArea
      case "nearestexact" | "nearestExact" | "kNearestExact" | "KNearest|Exact" => new kNearest
      case "linear" | "Linear" | "kLinear" | "KLinear"                          => new kLinear
    }
    options.mode().put(InterpolateMode(nativeMode))

    val lVec = new LongVector(output_size*)
    options.size().put(LongVectorOptional(lVec))
    val dVec = new DoubleVector(scale_factor*)

    options.scale_factor().put(DoubleVectorOptional(dVec))
    options.align_corners().put(BoolOptional(align_corners))
    options.recompute_scale_factor().put(BoolOptional(recompute_scale_factor))

    options.antialias().put(antialias)
    val result = torchNative.interpolate(x.native, options)
    fromNative(result)
  }

  def grid_sample[D <: DType](
      x: Tensor[D],
      grid: Tensor[D],
      mode: String = "bilinear",
      padding_mode: String = "zeros",
      align_corners: Boolean = false
  ): Tensor[D] = {
    val options = GridSampleFuncOptions()
    val nativeMode = mode match {
      case "nearest" | "Nearest"   => new kNearest
      case "bilinear" | "Bilinear" => new kBilinear
    }
    options.mode().put(GridSampleMode(nativeMode))
    val nativePaddingMode = padding_mode match {
      case "zeros" | "Zeros"           => new kZeros
      case "border" | "Border"         => new kBorder
      case "reflection" | "Reflection" => new kReflection

    }
    options.padding_mode().put(nativePaddingMode)
    options.align_corners().put(BoolOptional(align_corners))
    val result = torchNative.grid_sample(x.native, grid.native, options)
    fromNative(result)
  }

  def affine_grid[D <: DType](
      theta: Tensor[D],
      size: List[Long],
      align_corners: Option[Boolean] = None
  ): Tensor[D] = {

    val longVecRef = LongArrayRef(size.toArray, size.length)
    val result = align_corners match {
      case Some(s) => torchNative.affine_grid(theta.native, longVecRef, s)
      case None    => torchNative.affine_grid(theta.native, longVecRef)
    }
    fromNative(result)
  }

  def is_tensor(obj: AnyRef): Boolean = obj.isInstanceOf[Tensor[?]]

  def is_storage(obj: AnyRef): Boolean = obj.isInstanceOf[Storage]

  def init_num_threads = torchNative.init_num_threads()

  def get_num_threads = torchNative.get_num_threads()

  def get_thread_num = torchNative.get_thread_num()

  def divup(v1: Long, v2: Long): Long = torchNative.divup(v1, v2)

  def to_padded_tensor[D <: DType](input: Tensor[D], pad: Double): Tensor[D] = fromNative(
    torchNative.to_padded_tensor(input.native, pad)
  )

  def to_padded_tensor[D <: DType](input: Tensor[D], pad: Double, padVec: Array[Long]): Tensor[D] =
    fromNative(torchNative.to_padded_tensor(input.native, pad, padVec*))

  def synchronize = torchNative.synchronize()

  def commit = torchNative.commit()

  def get_command_buffer = torchNative.get_command_buffer()

  def get_dispatch_queue = torchNative.get_dispatch_queue()

  def numel[D <: DType](tensor: Tensor[D]): Long = torchNative.numel(tensor.native)

  object nested {

    def nested_tensor[D <: DType](tensorSeq: Seq[Tensor[D]]): Tensor[D] = fromNative(
      torchNative.nested_tensor(new TensorVector(tensorSeq.map(_.native)*))
    )

    def nested_tensor[D <: DType](
        tensorSeq: Seq[Tensor[D]],
        layout: Layout = Strided,
        device: Device = CPU,
        requiresGrad: Boolean = false
    ): Tensor[D] =
      val options = NativeConverters.tensorOptions(tensorSeq.head.dtype, layout, CPU, requiresGrad)
      fromNative(torchNative.nested_tensor(new TensorVector(tensorSeq.map(_.native)*), options))

    def as_nested_tensor[D <: DType](tensorSeq: Seq[Tensor[D]]): Tensor[D] = fromNative(
      torchNative.as_nested_tensor(new TensorVector(tensorSeq.map(_.native)*))
    )

    def as_nested_tensor[D <: DType](
        tensorSeq: Seq[Tensor[D]],
        dtype: DType,
        device: Device
    ): Tensor[D] = {
      val scalar = new ScalarTypeOptional(dtype.toScalarType)
      val deviceOpt = new DeviceOptional(device.toNative)
      fromNative(
        torchNative.as_nested_tensor(new TensorVector(tensorSeq.map(_.native)*), scalar, deviceOpt)
      )
    }
  }

  def broadcast_coalesced[D <: DType](
      process: ProcessGroup,
      tensors: Seq[Tensor[D]],
      buffer_size: Long
  ): Unit =
    torchNative.broadcast_coalesced(process, new TensorVector(tensors.map(_.native)*), buffer_size)

  def broadcast_coalesced[D <: DType](
      process: ProcessGroup,
      tensors: Seq[Tensor[D]],
      buffer_size: Long,
      size: Int
  ): Unit =
    torchNative.broadcast_coalesced(
      process,
      new TensorVector(tensors.map(_.native)*),
      buffer_size,
      size
    )

  def deviceCount = torchNative.deviceCount()

  def getDeviceIndex = torchNative.getDeviceIndex()

  def storage_copy(storage: Storage, storage2: Storage) =
    torchNative.storage_copy(storage, storage2)

  def share_memory_[D <: DType](tensor: Tensor[D]) = torchNative.share_memory_(tensor.native)

  def setCurrentStream(stream: Stream) = torchNative.setCurrentStream(stream)

  def getCurrentStream(b: Byte): Stream = torchNative.getCurrentStream(b)

  def getAccelerator(acc: Option[Boolean] = None) =
    if acc.isDefined then torchNative.getAccelerator(acc.get) else torchNative.getAccelerator()

  def register_privateuse1_backend(name: String) = torchNative.register_privateuse1_backend(name)

  def ExcludeFileExtension(ext: String) = torchNative.ExcludeFileExtension(ext)

  def StripBasename(name: String) = torchNative.StripBasename(name)

  def set_warning_handler(handler: WarningHandler) = torchNative.set_warning_handler(handler)

  def get_warnAlways: Boolean = torchNative.get_warnAlways()

  def set_warnAlways(always: Boolean) = torchNative.set_warnAlways(always)

  def get_warning_handler = torchNative.get_warning_handler()

  def schema(func: FunctionSchema) = torchNative.schema(func)

  def schema(schema: String, bool: Boolean) = torchNative.schema(schema, bool)

  def schema(schema: String) = torchNative.schema(schema)

  def setAutogradFallbackMode(mode: Int) = torchNative.setAutogradFallbackMode(mode)

  def autogradNotImplementedFallback = torchNative.autogradNotImplementedFallback()

  def autogradNotImplementedInplaceOrViewFallback =
    torchNative.autogradNotImplementedInplaceOrViewFallback()

  def basicAutogradNotImplementedFallback = torchNative.basicAutogradNotImplementedFallback()

  def getAutogradFallbackMode = torchNative.getAutogradFallbackMode()

  def getCopyOfFuncTorchTLS = torchNative.getCopyOfFuncTorchTLS()

  def functorchTLSAccessor = torchNative.functorchTLSAccessor()

  def torch_function_mode_enabled: Boolean = torchNative.torch_function_mode_enabled()

  def torch_function_all_disabled: Boolean = torchNative.torch_function_all_disabled()

  def dispatch_mode_enabled: Boolean = torchNative.dispatch_mode_enabled()

  def enter_dual_level = torchNative.enter_dual_level()

  def exit_dual_level(level: Long) = torchNative.exit_dual_level(level)

  def backward[D <: DType](tensors: Seq[Tensor[D]] | Tensor[D]) =
    val tensorsSeq = tensors match {
      case t: Tensor[D]       => Seq(t)
      case ts: Seq[Tensor[D]] => ts
    }
    torchNative.backward(new TensorVector(tensorsSeq.map(_.native)*))

  def grad[D1 <: DType, D2 <: DType](
      tensors: Seq[Tensor[D1]] | Tensor[D1],
      tensors2: Seq[Tensor[D2]] | Tensor[D2]
  ) =
    val tensorsSeq = tensors match {
      case t: Tensor[D1]       => Seq(t)
      case ts: Seq[Tensor[D1]] => ts
    }
    val tensors2Seq = tensors2 match {
      case t: Tensor[D2]       => Seq(t)
      case ts: Seq[Tensor[D2]] => ts
    }
    torchNative.grad(
      new TensorVector(tensorsSeq.map(_.native)*),
      new TensorVector(tensors2Seq.map(_.native)*)
    )

  def SetVariableHooks(hooks: VariableHooksInterface) = torchNative.SetVariableHooks(hooks)

  def GetVariableHooks = torchNative.GetVariableHooks()

  def HasVariableHooks: Boolean = torchNative.HasVariableHooks()

  def setDebugLevel(level: Int) = torchNative.setDebugLevel(level)

  def setDebugLevelFromEnvironment = torchNative.setDebugLevelFromEnvironment()

  def debug_level = torchNative.debug_level()

  def wait_tensor[D <: DType](tensor: Tensor[D]) = fromNative(
    torchNative.wait_tensor(tensor.native)
  )

  def register_work[D <: DType](tensor: Tensor[D], work: Work) =
    torchNative.register_work(tensor.native, work)

  def unregister_work(work: Work): Unit = torchNative.unregister_work(work)

  def get_work_registry_size(): Long = torchNative.get_work_registry_size()
}

//  def unique_consecutive[D <: DType](tensor: Tensor[D]) = torchNative.unique_consecutive(tensor.native)

//  def isAccelerator(deviceType: torch.DeviceType):Boolean = torchNative.isAccelerator(deviceType.toNative)
//  def isAcceleratorExcluded(deviceType: torch.DeviceType, deviceType2: torch.DeviceType):Boolean = torchNative.isAcceleratorExcluded(deviceType, deviceType2)
//

// def isProfilerEnabledInMainThread: Boolean = torchNative.isProfilerEnabledInMainThread()
// def enableProfilerInChildThread = torchNative.enableProfilerInChildThread()
//   def disableProfilerInChildThread = torchNative.disableProfilerInChildThread()

//  def dist[D <: DType](tensor: Tensor[D], tensor2: Tensor[D]): Tensor[D] =
//    fromNative(torchNative.dist(tensor.native, tensor2.native))

//    def einsum[D <:DType](
//        equation: String,
//        tensors: Tensor[D]*
//    ): Tensor[D] =
//      fromNative(torchNative.einsum(equation, TensorVector(tensors.map(_.native): _*)))
