package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
class ProductNeuralNetworkModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropout: Float,
    method: String = "inner"
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val numFields = fieldDims.size
  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val linear = register(nns.FeaturesLinear(fieldDims, embedDim))
  val embedOutputDim = numFields * embedDim
  val mlp1Dim = Math.floorDiv(numFields * (numFields - 1), 2) + embedOutputDim
  val mlp = register(nns.MultiLayerPerceptron(mlp1Dim, mlpDims, dropout))

  val pnInner = register(nns.InnerProductNetwork())
  val pnOuter = register(nns.OuterProductNetwork(numFields, embedDim))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val embed_x = this.embedding(x)
    if (method == "inner") {
      val cross_term = this.pnInner(embed_x)
      val embed_x_view = embed_x.view(-1, embedOutputDim)
      x = torch.cat(Seq(embed_x_view, cross_term), dim = 1)
    } else if (method == "outer") {
      val cross_term = this.pnOuter(embed_x)
      val embed_x_view = embed_x.view(-1, embedOutputDim)
      x = torch.cat(Seq(embed_x_view, cross_term), dim = 1)
    } else {}
    x = this.mlp(x)
    torch.sigmoid(x.squeeze(1)).to(x.dtype)

  }

}

object ProductNeuralNetworkModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropout: Float,
      method: String = "inner"
  ): ProductNeuralNetworkModel[ParamType] =
    new ProductNeuralNetworkModel(field_dims, embed_dim, mlp_dims, dropout, method)
