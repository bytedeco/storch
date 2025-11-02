package torch.utils.models.llm.moe

import torch.nn.modules.{HasParams, TensorModule}
import torch.*

class SparseMOE[ParamType <: FloatNN: Default](config: MOEConfig)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val hiddenDim: Int = config.hiddenDim
  val expertNumber: Int = config.expertNumber
  val topK: Int = config.topK
  val experts =
    nn.ModuleList((0 until expertNumber).map(num => new BasicExpert(hiddenDim, hiddenDim))*)
  val router = MOERouter(hiddenDim, expertNumber, topK)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def forward(x: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
    // x 形状是 (b, s, hiddenDim)
    val batchSize = x.shape(0).toInt
    val seqLen = x.shape(1).toInt
    // 合并前两个维度，因为不是Sample维度了，而是token维度
    val hiddenStates = x.view(-1, hiddenDim) // 形状是(b * s, hiddenDim)
    val (routerLogits, routerWeights, selectedExpertsIndices, expertMask) =
      router.forward(hiddenStates)
    // 创建输出张量
    val finalHiddenStates = torch.zeros(
      Seq(batchSize * seqLen, hiddenDim),
      dtype = hiddenStates.dtype,
      device = hiddenStates.device
    )
    // 对每个专家进行处理
    //    hiddenStates.toArray
    for (expertIdx <- 0 until expertNumber) {
      val expertLayer = experts(expertIdx)
      // 获取当前专家的掩码并找到需要处理的token
      val idx_topx = torch.where(expertMask(expertIdx))
      val idx = idx_topx(0)
      val topX: Tensor[ParamType] = idx_topx(1)
      //      topX.numpy().toArray
      //      val (idx, topX) = torch.where(expertMask(expertIdx))
      val hiddenStateUnsqueezed = hiddenStates.unsqueeze(0)
      // 提取需要处理的token的隐藏状态
      //      val currentState = hiddenStateUnsqueezed(::,topX.toArray.toSeq.asInstanceOf[Seq[Long]], ::).reshape(-1, hiddenDim)
      val currentState = hiddenStateUnsqueezed(::, topX.to(DType.int64), ::).reshape(-1, hiddenDim)
      // 应用专家层并加权
      val weights = routerWeights(topX.to(DType.int64), idx.to(DType.int64)).unsqueeze(-1)
      val currentHiddenStates = expertLayer(currentState) * weights

      // 将当前专家的输出加到最终结果中
      finalHiddenStates.index_add_(
        0,
        topX.to(DType.int64),
        currentHiddenStates.to(hiddenStates.dtype)
      )
    }
    // 将结果还原到原始形状
    val reshapedOutput = finalHiddenStates.reshape(batchSize, seqLen, hiddenDim)

    (reshapedOutput, routerLogits)
  }
}
