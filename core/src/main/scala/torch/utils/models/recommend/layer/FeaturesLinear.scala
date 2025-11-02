package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class FeaturesLinear[ParamType <: FloatNN: Default](fieldDim: Seq[Int], outputDim: Int = 1)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nn.Embedding(fieldDim.sum, outputDim))
  val bias = torch.zeros(Seq(outputDim, fieldDim.last))
  val offsets =
    fieldDim.map(_.toInt).scanLeft(0)((ep, et) => ep + et).dropRight(1).map(_.toLong).toArray

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val offsetsTensor = torch.Tensor(offsets).unsqueeze(0)
    input.add(offsetsTensor) // .to[torch.Int64]
    val sumTensor = torch.sum(embedding(input))
//    val sumTensor = torch.sum(embedding(input.to(dtype = torch.int64)))
    val out = sumTensor.add(bias) // .to(dtype = this.paramType)
    out.to(dtype = this.paramType)

  }

}

//    val embOut = fc(input.to(dtype = this.paramType))
//    torch.sum(input= embOut.to(dtype = this.paramType),dim =1, keepdim =false).add(bias)

object FeaturesLinear:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      output_dims: Int = 1
  ): FeaturesLinear[ParamType] = new FeaturesLinear(field_dims, output_dims)
