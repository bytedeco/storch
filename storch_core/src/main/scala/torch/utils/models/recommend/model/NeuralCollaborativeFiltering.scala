package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class NeuralCollaborativeFiltering[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    user_field_idx: Int,
    item_field_idx: Int,
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val embedOutputDim = fieldDims.size * embedDim
  val mlp = register(nns.MultiLayerPerceptron(embedOutputDim, mlpDims, dropout, false))
  val fc = register(nn.Linear(mlpDims.last + embedDim, 1))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = this.embedding(input)
    val user_x = x.index(::, user_field_idx)
    val item_x = x.index(::, item_field_idx)
    x = this.mlp(x.view(-1, embedOutputDim))
    val gmf = torch.dot(user_x.squeeze(1), item_x.squeeze(1))
    x = torch.cat(Seq(gmf, x), 1)
    x = this.fc(x).squeeze(1)
    torch.sigmoid(x).to(x.dtype)

  }

}

object NeuralCollaborativeFiltering:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      user_field_idx: Int,
      item_field_idx: Int,
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): NeuralCollaborativeFiltering[ParamType] =
    new NeuralCollaborativeFiltering(
      field_dims,
      user_field_idx,
      item_field_idx,
      embed_dim,
      mlp_dims,
      dropout
    )
