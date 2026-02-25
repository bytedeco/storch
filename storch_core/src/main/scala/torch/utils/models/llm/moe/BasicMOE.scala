package torch.utils.models.llm.moe

import torch.nn.Linear
import torch.nn.modules.{HasParams, TensorModule}
import torch.{Default, FloatNN, Tensor, nn}

class BasicMOE[ParamType <: FloatNN: Default](featureIn: Int, featureOut: Int, expertNumber: Int)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  val experts =
    nn.ModuleList((0 until expertNumber).map(num => new BasicExpert(featureIn, featureOut))*)
  val gate = register(Linear(featureIn, expertNumber))
  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
    // x 的形状是 (batch, featureIn)
    val expertWeight = gate.forward(x) // 形状是 (batch, expertNumber)
    // 计算每个专家的输出并增加一个维度
    val expertOutList = experts.map(expert => expert(x).unsqueeze(1))
    // 拼接专家输出，形状变为 (batch, expertNumber, featureOut)
    val expertOutput = torch.cat(expertOutList.toSeq, dim = 1)
    // 调整权重形状以进行矩阵乘法
    val reshapedWeight = expertWeight.unsqueeze(1) // (batch, 1, expertNumber)

    // 矩阵乘法计算最终输出
    val output = reshapedWeight.matmul(expertOutput) // (batch, 1, featureOut)

    // 移除多余的维度
    output.squeeze()
  }
}
