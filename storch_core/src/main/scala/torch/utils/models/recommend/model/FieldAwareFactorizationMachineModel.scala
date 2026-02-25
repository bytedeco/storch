package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
class FieldAwareFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nns.FeaturesLinear(fieldDims, 1))
  val ffm = register(nns.FieldAwareFactorizationMachine(fieldDims, embedDim))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var x = input
    val tmp_term = torch.sum(this.ffm(x), dim = 1)
    val ffm_term = torch.sum(tmp_term, dim = 1, keepdim = true)
    x = this.linear(x).add(ffm_term)
    torch.sigmoid(x.squeeze(1)).to(x.dtype)
  }

}

object FieldAwareFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int
  ): FieldAwareFactorizationMachineModel[ParamType] =
    new FieldAwareFactorizationMachineModel(field_dims, embed_dim)
