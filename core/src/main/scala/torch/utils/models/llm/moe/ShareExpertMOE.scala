package torch.utils.models.llm.moe

import torch.nn.modules.{HasParams, TensorModule}
import torch.{Default, FloatNN, Tensor, nn}

class ShareExpertMOE[ParamType <: FloatNN: Default](config: MOEConfig)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val moeModel = SparseMOE(config)
  val sharedExperts = nn.ModuleList(
    (0 until config.sharedExpertsNumber).map(num =>
      BasicExpert(config.hiddenDim, config.hiddenDim)
    )*
  )
  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
  def forward(x: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    // 首先通过moe模型
    val (sparseMoeOut, routerLogits) = moeModel.forward(x)
    // 然后通过共享专家
    val sharedExpertsOut = sharedExperts.map(expert => expert(x))
    // 堆叠共享专家的输出并求和
    val sharedExpertsOutSum = torch.stack(sharedExpertsOut.toSeq, dim = 0).sum(dim = 0)
    // 将sparse_moe_out和shared_experts_out相加
    (sparseMoeOut + sharedExpertsOutSum, routerLogits)
  }
}
