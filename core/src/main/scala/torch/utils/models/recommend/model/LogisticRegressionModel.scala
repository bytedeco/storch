package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class LogisticRegressionModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int]
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nns.FeaturesLinear(fieldDims, 1))

  def apply(input: Tensor[ParamType]): Tensor[ParamType] =
    torch.sigmoid(linear(input).squeeze(1)).to(input.dtype)

}

object LogisticRegressionModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int]
  ): LogisticRegressionModel[ParamType] =
    new LogisticRegressionModel(field_dims)
