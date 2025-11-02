package torch.utils.models.speech.transformerASR

import scala.collection.mutable.ArrayBuffer
import org.apache.commons.math3.transform.{DftNormalization, FastFourierTransformer, TransformType}
import torch.nn
import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{Dropout, Embedding, Linear, Transformer, functional as F}
import torch.::
import torch.nn.loss.CTCLoss
import torch.optim.Adam
import torch.utils.data.{Dataset, DataLoader}
import java.nio.file.Paths
import scala.math.*

// CTC损失函数包装（适用于语音识别）
class ASRLoss[ParamType <: FloatNN: Default] extends TensorModule[ParamType]:
  private val ctcLoss = CTCLoss(blank = 0, reduction = "mean")

  def forward(
      logits: Tensor[ParamType], // (batch_size, tgt_len, vocab_size)
      targets: Tensor[ParamType], // (batch_size, target_len)
      inputLengths: Tensor[ParamType], // (batch_size,)
      targetLengths: Tensor[ParamType] // (batch_size,)
  ): Tensor[ParamType] =
    // CTC Loss要求logits形状为 (tgt_len, batch_size, vocab_size)
    val logitsTransposed = logits.permute(1, 0, 2).log_softmax(2)
    ctcLoss(logitsTransposed, targets, inputLengths, targetLengths)

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(
      logits: Tensor[ParamType],
      targets: Tensor[ParamType],
      inputLengths: Tensor[ParamType],
      targetLengths: Tensor[ParamType]
  ): Tensor[ParamType] = forward(logits, targets, inputLengths, targetLengths)
