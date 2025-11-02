package torch.utils.models.llm.moe

import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.*

class MOERouter[ParamType <: FloatNN: Default](hiddenDim: Int, expertNumber: Int, topK: Int)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val gate = register(nn.Linear(hiddenDim, expertNumber))

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def forward(
      hiddenStates: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType], Tensor[Int64], Tensor[ParamType]) = {
    // 计算路由logits
    val routerLogits = gate.forward(hiddenStates) // 形状是 (b * s, expertNumber)
    // 计算专家经过softmax之后的概率
    val routingProbs = F.softmax(routerLogits, dim = -1, dtype = hiddenStates.dtype)
    // 计算topk的专家的输出
    val routerWeights_selectedExperts =
      torch.topk(routingProbs, topK, dim = -1) // , largest = true, sorted = true
    val (routerWeights, selectedExperts) =
      (routerWeights_selectedExperts._1, routerWeights_selectedExperts._2)
    // 专家权重归一化
    val normalizedWeights = routerWeights / routerWeights.sum(dim = -1, keepdim = true)
    val routerWeightsTyped = normalizedWeights.to(hiddenStates.dtype)
    // 生成专家掩码
    val expertMask = F.one_hot(selectedExperts, numClasses = expertNumber)
    val permutedMask = expertMask.permute(2, 1, 0).to(hiddenStates.dtype)
    (routerLogits, routerWeightsTyped, selectedExperts, permutedMask)
  }
}
