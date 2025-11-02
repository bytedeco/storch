package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
class AttentionalFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    attnSize: Int,
    dropouts: Seq[Float]
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val numFields = fieldDims.size
  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val linear = register(nns.FeaturesLinear(fieldDims, 1))
  val afm = register(nns.AttentionalFactorizationMachine(embedDim, attnSize, dropouts))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    x = this.linear(x).add(this.afm(this.embedding(x)))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object AttentionalFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      attn_size: Int,
      dropouts: Seq[Float]
  ): AttentionalFactorizationMachineModel[ParamType] =
    new AttentionalFactorizationMachineModel(field_dims, embed_dim, attn_size, dropouts)
