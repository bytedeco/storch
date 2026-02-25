package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
class NeuralFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropouts: Seq[Float]
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val FMBnDropBlockSeq = nn.Sequential(
    nn.BatchNorm1d(embedDim),
    nn.Dropout(dropouts.head),
    nns.FactorizationMachine(reduce_sum = false)
  )
  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim))
  val linear = register(nns.FeaturesLinear(fieldDims))
//  val fm = register(FMBnDropBlockSeq())
  val mlp = register(nns.MultiLayerPerceptron(embedDim, mlpDims, dropouts(1)))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val cross_term = this.FMBnDropBlockSeq(this.embedding(x))
    x = this.linear(x).add(this.mlp(cross_term))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object NeuralFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropouts: Seq[Float]
  ): NeuralFactorizationMachineModel[ParamType] =
    new NeuralFactorizationMachineModel(field_dims, embed_dim, mlp_dims, dropouts)
