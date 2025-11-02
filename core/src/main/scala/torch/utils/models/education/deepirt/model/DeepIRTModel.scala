package torch.utils.models.education.deepirt.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{Embedding, Linear, Module, Parameter, functional as F}
import torch.nn.{Module, Parameter, functional as F}
import torch.utils.models.education.deepirt.layer.DeepIRT

import scala.collection.mutable.ArrayBuffer

class DeepIRTModel[ParamType <: FloatNN: Default](
    val n_question: Int,
    val batch_size: Int,
    val q_embed_dim: Int,
    val qa_embed_dim: Int,
    val memory_size: Int,
    val final_fc_dim: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  // 线性层定义
  val read_embed_linear = nn.Linear(q_embed_dim + qa_embed_dim, final_fc_dim, bias = true)
  val predict_linear = nn.Linear(final_fc_dim, 1, bias = true)
  val beta_linear = nn.Linear(q_embed_dim, 1, bias = true)

  // 内存初始化参数
  val memory_key: Tensor[ParamType] =
    torch.randn(Seq(memory_size, q_embed_dim), dtype = this.paramType)
  val memory_value: Tensor[ParamType] =
    torch.randn(Seq(memory_size, qa_embed_dim), dtype = this.paramType)

  val init_memory_key = nn.Parameter[ParamType](
    name = "init_memory_key",
    weight = memory_key,
    requires_grad = true
  ) // Mk的emb
  val init_memory_value = nn.Parameter[ParamType](
    name = "init_memory_value",
    weight = memory_value,
    requires_grad = true
  ) // Ntv的emb

  // 初始化DeepIRT模块
  val mem = DeepIRT[ParamType](
    memorySize = memory_size,
    memoryKeyStateDim = q_embed_dim,
    memoryValueStateDim = qa_embed_dim,
    initMemoryKey = init_memory_key
  )

  // 嵌入层
  val q_embed = nn.Embedding(n_question + 1, q_embed_dim, padding_idx = 0)
  val qa_embed = nn.Embedding(2 * n_question + 1, qa_embed_dim, padding_idx = 0)

  // 初始化参数
  init_params()
  init_embeddings()

  def init_params(): Unit = {
    nn.init.kaimingNormal_(read_embed_linear.weight)
    nn.init.kaimingNormal_(predict_linear.weight)
    nn.init.kaimingNormal_(beta_linear.weight)

    nn.init.constant_(read_embed_linear.bias, 0)
    nn.init.constant_(predict_linear.bias, 0)
    nn.init.constant_(beta_linear.bias, 0)

    nn.init.kaimingNormal_(init_memory_key)
    nn.init.kaimingNormal_(init_memory_value)
  }

  def init_embeddings(): Unit = {
    nn.init.kaimingNormal_(q_embed.weight)
    nn.init.kaimingNormal_(qa_embed.weight)
  }

  def forward(
      q_data: Tensor[ParamType],
      qa_data: Tensor[ParamType],
      target: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {

    val batch_size = q_data.shape(0)
    val seqlen = q_data.shape(1)

    // 获取嵌入
    val q_embed_data = q_embed(q_data)
    val qa_embed_data = qa_embed(qa_data)

    // 初始化内存值
    val memory_value = init_memory_value.unsqueeze(0).repeat(batch_size, 1, 1)
    mem.init_value_memory(memory_value)

    // 分割序列
    val slice_q_embed_data = q_embed_data.split(1, dim = 1).map(_.squeeze(1))
    val slice_qa_embed_data = qa_embed_data.split(1, dim = 1).map(_.squeeze(1))

    val value_read_content_l = ArrayBuffer[Tensor[ParamType]]()
    val input_embed_l = ArrayBuffer[Tensor[ParamType]]()
    val q_embed_l = ArrayBuffer[Tensor[ParamType]]()

    for (i <- 0 until seqlen) {
      // Attention过程
      val q = slice_q_embed_data(i)
      val correlation_weight = mem.attention(q)
      q_embed_l += q

      // 读取过程
      val read_content = mem.read(readWeight = correlation_weight)
      value_read_content_l += read_content
      input_embed_l += q

      // 写入过程
      val qa = slice_qa_embed_data(i)
      mem.write(correlation_weight, qa)
    }

    // 拼接结果
    val all_read_value_content = torch.stack(value_read_content_l.toSeq, dim = 1)
    val input_embed_content = torch.stack(input_embed_l.toSeq, dim = 1)
    val q_embed_content = torch.stack(q_embed_l.toSeq, dim = 1).view(batch_size * seqlen, -1)

    // 预测输入
    val predict_input = torch.cat(Array(all_read_value_content, input_embed_content), dim = 2)
    val read_content_embed = F.tanh(read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

    // 计算预测值
    val seta = predict_linear(read_content_embed)
    val beta = F.tanh(beta_linear(q_embed_content))
    val pred = F.tanh(torch.tensor(0.3f) * seta - beta)

    // 计算损失
    val target_1d = target.view(-1, 1)
    val mask = target_1d.ge(1)
    val pred_1d = pred.view(-1, 1)

    val filtered_pred = pred_1d.masked_select(mask)
    val filtered_target = target_1d.masked_select(mask) - 1
    val loss = F.binaryCrossEntropyWithLogits(filtered_pred.to(torch.float32), filtered_target)
    (loss, F.sigmoid(filtered_pred.to(q_data.dtype)), filtered_target)
  }

  // 实现参数访问
  override def parameters: Seq[Tensor[ParamType]] = {
    read_embed_linear.parameters ++
      predict_linear.parameters ++
      beta_linear.parameters ++
      q_embed.parameters ++
      qa_embed.parameters ++
      mem.parameters ++
      Seq(init_memory_key, init_memory_value)
  }

  def apply(
      q_data: Tensor[ParamType],
      qa_data: Tensor[ParamType],
      target: Tensor[ParamType]
  ): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = forward(q_data, qa_data, target)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}
