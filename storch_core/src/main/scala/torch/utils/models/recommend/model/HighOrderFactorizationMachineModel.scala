package torch.utils.models.recommend.model

import torch.*
import torch.nn.modules.container.ModuleList
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString

class HighOrderFactorizationMachineModel[ParamType <: FloatNN: Default](
    fieldDims: Seq[Int],
    embedDim: Int,
    order: Int,
    mlpDims: Seq[Int],
    dropout: Float
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val linear = register(nns.FeaturesLinear(fieldDims))
  val fm = register(nns.FactorizationMachine(reduce_sum = true))
  val embedding = register(nns.FeaturesEmbedding(fieldDims, embedDim * (order - 1)))
  val kernels = ModuleList[ParamType]()
  if (order >= 3) {
    for (i <- 3 to order + 1 by 1) {
      val anovaKernel = register(nns.AnovaKernel(i, true))
      kernels.append(anovaKernel)

    }
  }

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    val y = linear(input).squeeze(1)
    if (order >= 2) {
      val in = embedding(input)
      var x_part = in.index(::, ::, 0.&&(embedDim))
      y += fm(x_part).squeeze(1)
      for (i <- 0 until (order - 2) by 1) {
        val dim_3_start = (i + 1) * embedDim
        val dim_3_end = (i + 1) * embedDim
        x_part = in.index(::, ::, dim_3_start.&&(dim_3_end))
        y += kernels(i)(x_part).squeeze(1)
      }
    }
    torch.sigmoid(y).to(dtype = this.paramType)
  }
}

object HighOrderFactorizationMachineModel:
  def apply[ParamType <: FloatNN: Default](
      field_dims: Seq[Int],
      embed_dim: Int,
      order: Int,
      mlp_dims: Seq[Int],
      dropout: Float
  ): HighOrderFactorizationMachineModel[ParamType] =
    new HighOrderFactorizationMachineModel(field_dims, embed_dim, order, mlp_dims, dropout)

//    val out = torch.sigmoid(x.squeeze(1).to(dtype = this.paramType))
//    out.to(dtype = this.paramType)
