package torch
package nn
package modules
package normalization

import org.bytedeco.javacpp.{LongPointer, DoublePointer, BoolPointer}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{LayerNormImpl, LayerNormOptions, LongVector}
import internal.NativeConverters.fromNative

final class RMSNorm[ParamType <: FloatNN | ComplexNN: Default](
    val normalized_shape: Seq[Int] | Int,
    val eps: Double = 1e-05,
    val elementwise_affine: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")

  private val normalizedShapeLong: Seq[Long] = normalized_shape match
    case s: Seq[Int] => s.map(_.toLong)
    case i: Int      => Seq(i).map(_.toLong)
  var weight: Tensor[ParamType] =
    registerParameter(
      torch.empty(normalizedShapeLong.map(_.toInt), dtype = this.paramType, requires_grad = true),
      true,
      "weight"
    )

  override def hasBias(): Boolean = false

  override def toString =
    s"${getClass.getSimpleName}(normalizedShape = ${normalizedShapeLong.mkString(" ")}, elementWiseAffine = ${elementwise_affine},eps=$eps )"

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val output =
      if elementwise_affine then
        torch.rms_norm(
          input,
          normalizedShapeLong,
          Some(weight.to(DType.float32)),
          Some(eps)
        )
      else torch.rms_norm(input, normalizedShapeLong, None, Some(eps))
    output.to(input.dtype)
  }

  def forward(input: Tensor[ParamType]): Tensor[ParamType] = {
    val output =
      if elementwise_affine then
        torch.rms_norm(
          input,
          normalizedShapeLong,
          Some(weight.to(DType.float32)),
          Some(eps)
        )
      else torch.rms_norm(input, normalizedShapeLong, None, Some(eps))
    output.to(input.dtype)
  }

object RMSNorm:
  def apply[ParamType <: FloatNN | ComplexNN: Default](
      normalized_shape: Seq[Int] | Int,
      eps: Double = 1e-05,
      elementwise_affine: Boolean = true
  ): RMSNorm[ParamType] =
    new RMSNorm(normalized_shape, eps, elementwise_affine)
