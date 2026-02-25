package torch
package utils
package trainer
import torch.nn as nn
import torch.nn.functional as F
import torch.*
import torch.nn.modules.*

class LstmNet[D <: BFloat16 | FloatNN: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D]
    with TensorModule[D] {

  val lstm = register(nn.LSTM(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))

  def apply(input: Tensor[D]): Tensor[D] = forward(input)

  def forward(input: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, input.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = input.dtype)
    val c0 = torch.zeros(size = arr, dtype = input.dtype)
    val outTuple3 = lstm(input, Some(h0), Some(c0))
    var out: Tensor[D] = outTuple3._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}

class RnnNet[D <: BFloat16 | FloatNN: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D] {

  val rnn = register(nn.RNN(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))

  def apply(i: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = i.dtype)
    val c0 = torch.zeros(size = arr, dtype = i.dtype)
    val outTuple2 = rnn(i, Some(h0))
    var out: Tensor[D] = outTuple2._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}

class GruNet[D <: BFloat16 | FloatNN: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D] {

  val gru = register(nn.GRU(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))

  def apply(i: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = i.dtype)
    val c0 = torch.zeros(size = arr, dtype = i.dtype)
    val outTuple2 = gru(i, Some(h0))
    var out: Tensor[D] = outTuple2._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}
