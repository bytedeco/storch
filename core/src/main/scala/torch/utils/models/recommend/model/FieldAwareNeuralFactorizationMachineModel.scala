package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class FieldAwareNeuralFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    mlpDims: Seq[Int],
    dropouts: Seq[Float]
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nns.FeaturesLinear(fieldDims))
  val ffm = register(nns.FieldAwareFactorizationMachine(fieldDims, embedDim))
  val ffmOutputDim = Math.floorDiv(fieldDims.size * (fieldDims.size - 1), 2) * embedDim
  val mlp = register(nns.MultiLayerPerceptron(ffmOutputDim, mlpDims, dropouts(1)))
  val dropout = register(nn.Dropout(dropouts.head))
  val bn = register(nn.BatchNorm1d(ffmOutputDim))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    var cross_term = this.ffm(x).view(-1, ffmOutputDim)
    cross_term = this.bn(cross_term)
    cross_term = this.dropout(cross_term)
    x = this.linear(x).add(this.mlp(cross_term))
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }
}

object FieldAwareNeuralFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      mlp_dims: Seq[Int],
      dropouts: Seq[Float]
  ): FieldAwareNeuralFactorizationMachineModel[ParamType] =
    new FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim, mlp_dims, dropouts)
