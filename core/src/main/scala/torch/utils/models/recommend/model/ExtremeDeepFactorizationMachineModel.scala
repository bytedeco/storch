package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
class ExtremeDeepFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float,
    crossLayerSizes: Seq[Int],
    splitHalf: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val linear = register(nns.FeaturesLinear(fieldDims, 1))
  val mlp = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropout))
  val ciNet = register(nns.CompressedInteractionNetwork(fieldDims.size, crossLayerSizes, splitHalf))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x)
    x = this.linear(x).add(this.ciNet(embed_x)).add(this.mlp(embed_x.view(-1, embedOutputDim)))
    torch.sigmoid(x.squeeze(1)).to(input.dtype)
  }

}

object ExtremeDeepFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float,
      cross_layer_sizes: Seq[Int],
      split_half: Boolean = true
  ): ExtremeDeepFactorizationMachineModel[ParamType] =
    new ExtremeDeepFactorizationMachineModel(
      field_dims,
      embed_dim,
      mlp_dims,
      dropout,
      cross_layer_sizes,
      split_half
    )
