package torch.utils.models.recommend.model

import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class FactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val linear = register(nns.FeaturesLinear(fieldDims, 1))
  val fm = register(nns.FactorizationMachine(reduce_sum = true))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    x = this.linear(x).add(this.fm(this.embedding(x)))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object FactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int
  ): FactorizationMachineModel[ParamType] =
    new FactorizationMachineModel(field_dims, embed_dim)
