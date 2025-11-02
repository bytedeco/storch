package torch.utils.models.recommend.model

import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class FactorizationSupportedNeuralNetworkModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val mlp = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropout))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x)
    x = this.mlp(embed_x.view(-1, embedOutputDim))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object FactorizationSupportedNeuralNetworkModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): FactorizationSupportedNeuralNetworkModel[ParamType] =
    new FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim, mlp_dims, dropout)
