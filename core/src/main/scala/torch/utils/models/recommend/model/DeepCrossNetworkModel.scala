package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class DeepCrossNetworkModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    numLayers: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val crossNetworks = register(nns.CrossNetwork(embedOutputDim, numLayers))
  val mlp = register(
    nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropout, output_layer = false)
  )
  val linear = register(nn.Linear(embedOutputDim + mlpDims.last, 1))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x).view(-1, embedOutputDim)
    val x_l1 = this.crossNetworks(embed_x)
    val h_l2 = this.mlp(embed_x)
    val p = this.linear(torch.cat(Seq(x_l1, h_l2), dim = 1))
    torch.sigmoid(p.squeeze(1)).to(x.dtype)
  }

}

object DeepCrossNetworkModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      num_layers: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): DeepCrossNetworkModel[ParamType] =
    new DeepCrossNetworkModel(field_dims, embed_dim, num_layers, mlp_dims, dropout)
