package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class ConvBnReluBlock[ParamType <: FloatNN: Default](
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val conv = register(nn.Conv2d(inChannels, outChannels, kernelSize, stride))
  val bn = register(nn.BatchNorm2d(outChannels))
  val relu = register(nn.ReLU())

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val out = conv(input.to(dtype = this.paramType))
    relu(bn(out))
  }
}

object ConvBnReluBlock:
  def apply[ParamType <: FloatNN: Default](
      in_channels: Int,
      out_channels: Int,
      kernel_size: Int,
      stride: Int
  ): ConvBnReluBlock[ParamType] =
    new ConvBnReluBlock(in_channels, out_channels, kernel_size, stride)
