package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class WideAndDeepModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nns.FeaturesLinear(field_dims = fieldDims))
  val embedding = register(nns.FeaturesEmbedding(field_dims = fieldDims, embed_dim = embedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val mlp = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropout))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x)
    x = this.linear(x).add(this.mlp(embed_x.view(-1, embedOutputDim)))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }
}

object WideAndDeepModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): WideAndDeepModel[ParamType] =
    new WideAndDeepModel(field_dims, embed_dim, mlp_dims, dropout)
