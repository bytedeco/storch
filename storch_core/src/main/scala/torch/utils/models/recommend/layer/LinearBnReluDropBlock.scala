package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class LinearBnReluDropBlock[ParamType <: FloatNN: Default](
    inputDim: Int,
    embedDim: Int,
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nn.Linear(inputDim, embedDim))
  val relu = register(nn.ReLU())
  val dropoutLayer = register(nn.Dropout(dropout))
  val bn = register(nn.BatchNorm2d(embedDim))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    val out = linear(input.to(dtype = this.paramType))
    dropoutLayer(relu(bn(out)))
  }

}

object LinearBnReluDropBlock:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Int,
      embed_dim: Int,
      dropout: Float
  ): LinearBnReluDropBlock[ParamType] = new LinearBnReluDropBlock(input_dim, embed_dim, dropout)

//    val out = linear.forward(input)
//    val out = bn.forward(out)
//    val out = relu.forward(out)
//    val out = dropout.forward(out)
//    out
//    val out1 = linear(input.to(dtype = torch.DType))
