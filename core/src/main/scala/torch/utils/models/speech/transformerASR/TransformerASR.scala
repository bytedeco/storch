package torch.utils.models.speech.transformerASR

//import org.apache.commons.math3.transform.{DftNormalization, FastFourierTransformer, TransformType}
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{Dropout, Embedding, Linear, Transformer}

// 位置编码层实现
class PositionalEncoding[ParamType <: FloatNN: Default](
    val d_model: Int,
    val dropout: Double,
    val max_len: Int = 5000
) extends HasParams[ParamType]
    with TensorModule[ParamType]:

  // 初始化位置编码矩阵
  private val pe: Tensor[ParamType] = torch.zeros(Seq(max_len, d_model)).to(this.paramType)

  // 计算位置编码
  private val position = torch.arange(0, max_len).unsqueeze(1)
  private val div_term = torch.exp(
    torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
  )

  pe.slice(1, 0, d_model, 2).copy_(position * div_term).sin_()
  pe.slice(1, 1, d_model, 2).copy_(position * div_term).cos_()
  pe.unsqueeze_(0).transpose_(0, 1) // 形状: (max_len, 1, d_model)

  // 注册为非训练参数
  registerBuffer("pe", pe)

  // Dropout层
  val dropoutLayer: Dropout[ParamType] = nn.Dropout(dropout)

  // 前向传播
  def forward(x: Tensor[ParamType]): Tensor[ParamType] =
    // x形状: (seq_len, batch_size, d_model)
    val xWithPE = x + pe(0.::(x.size(0)), ::) // 截取对应长度的位置编码
    dropoutLayer(xWithPE)

  // apply方法
  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)

  // 伴生对象构造方法
  object PositionalEncoding:
    def apply[ParamType <: FloatNN: Default](
        d_model: Int,
        dropout: Double = 0.1,
        max_len: Int = 5000
    ): PositionalEncoding[ParamType] =
      new PositionalEncoding[ParamType](d_model, dropout, max_len)

// TransformerASR主模型
class TransformerASR[ParamType <: FloatNN: Default](
    val input_dim: Int,
    val vocab_size: Int,
    val d_model: Int = 256,
    val nhead: Int = 4,
    val num_encoder_layers: Int = 6,
    val num_decoder_layers: Int = 6,
    val dim_feedforward: Int = 1024,
    val dropout: Double = 0.1
) extends HasParams[ParamType]
    with TensorModule[ParamType]:

  // 输入投影层（将MFCC特征映射到d_model维度）
  val input_proj: Linear[ParamType] = nn.Linear(input_dim, d_model)

  // 位置编码器
  val pos_encoder: PositionalEncoding[ParamType] =
    PositionalEncoding[ParamType](d_model, dropout)

  // 目标序列嵌入层
  val tgt_embedding: Embedding[ParamType] =
    nn.Embedding(vocab_size, d_model)

  // Transformer模型（Storch实现）
  val transformer: Transformer[ParamType] = Transformer(
    d_model = d_model,
    nhead = nhead,
    num_encoder_layers = num_encoder_layers,
    num_decoder_layers = num_decoder_layers,
    dim_feedforward = dim_feedforward,
    dropout = dropout
  )

  // 输出层（映射到词汇表大小）
  val output_layer: Linear[ParamType] = Linear(d_model, vocab_size)

  // 参数初始化
  initParams()

  // 参数初始化方法（Xavier均匀分布）
  private def initParams(): Unit =
    for (param <- parameters) {
      if (param.dim > 1) nn.init.xavierUniform_(param)
    }

  // 前向传播
  def forward(
      src: Tensor[ParamType], // 输入特征: (batch_size, seq_len, input_dim)
      tgt: Tensor[ParamType], // 目标序列: (batch_size, tgt_len)
      src_key_padding_mask: Option[Tensor[ParamType]] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] =
    // 1. 输入投影 + 位置编码 (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
    val srcProj = input_proj(src).permute(1, 0, 2) // 转置为(seq_len, batch_size, d_model)
    val srcWithPE = pos_encoder(srcProj)

    // 2. 目标序列嵌入 + 位置编码 (batch_size, tgt_len) -> (tgt_len, batch_size, d_model)
    val tgtEmb = tgt_embedding(tgt).permute(1, 0, 2) // 转置为(tgt_len, batch_size, d_model)
    val tgtWithPE = pos_encoder(tgtEmb)

    // 3. Transformer前向传播
    val transformerOutput = transformer(
      src = srcWithPE,
      tgt = tgtWithPE,
      src_key_padding_mask = src_key_padding_mask,
      tgt_key_padding_mask = tgt_key_padding_mask,
      memory_key_padding_mask = src_key_padding_mask
    ) // (tgt_len, batch_size, d_model)

    // 4. 输出层映射 (tgt_len, batch_size, vocab_size) -> (batch_size, tgt_len, vocab_size)
    output_layer(transformerOutput).permute(1, 0, 2)

  // apply方法（调用forward）
  def apply(
      src: Tensor[ParamType],
      tgt: Tensor[ParamType],
      src_key_padding_mask: Option[Tensor[ParamType]] = None,
      tgt_key_padding_mask: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = forward(src, tgt, src_key_padding_mask, tgt_key_padding_mask)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
  // 伴生对象构造方法
object TransformerASR:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Int,
      vocab_size: Int,
      d_model: Int = 256,
      nhead: Int = 4,
      num_encoder_layers: Int = 6,
      num_decoder_layers: Int = 6,
      dim_feedforward: Int = 1024,
      dropout: Double = 0.1
  ): TransformerASR[ParamType] =
    new TransformerASR[ParamType](
      input_dim,
      vocab_size,
      d_model,
      nhead,
      num_encoder_layers,
      num_decoder_layers,
      dim_feedforward,
      dropout
    )
