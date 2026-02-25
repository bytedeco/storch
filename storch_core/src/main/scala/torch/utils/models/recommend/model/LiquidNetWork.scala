package torch.utils.models.recommend.model

import org.bytedeco.pytorch.GeneratorOptional
import torch.{Tensor, *}
import torch.nn.modules.{HasParams, TensorModule}
import torch.utils.models.recommend.layer as nns
import torch.utils.models.recommend.layer.IntOrString
import torch.nn.functional as fn
import org.bytedeco.pytorch.global.torch.*
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.pytorch.*
import torch.nn.modules.container.Sequential
//import torch.ops.GradOps

import scala.collection.mutable.ListBuffer

class LiquidNetWork[ParamType <: FloatNN: Default](
    inputSize: Int,
    hiddenSize: Int,
    numLayers: Int
) extends HasParams[ParamType]
    with TensorModule[ParamType] {
// 定义 LNN 类
//class LNNetwork[](inputSize: Long, hiddenSize: Long, numLayers: Int) extends Module {
  private val hiddenSizeVal = hiddenSize
  private val numLayersVal = numLayers
  val layers: ListBuffer[Sequential[ParamType]] = new ListBuffer[Sequential[ParamType]]()

  for (_ <- 0 until numLayers) {
    layers += createLayer(inputSize, hiddenSize)
    registerModule(layers.last, s"layer_${layers.size - 1}")
  }

  private def createLayer(inputSize: Int, hiddenSize: Int): Sequential[ParamType] = {
    val seq = Sequential(
      nn.Linear(inputSize, hiddenSize),
      nn.LeakyReLU(),
      nn.Linear(hiddenSize, hiddenSize)
    )
    seq
  }

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    var result = input
    layers.foreach(layer => layer(result))
    result
  }
}

// 定义 ODESolver 类
class ODESolver[ParamType <: FloatNN: Default](model: LiquidNetWork[ParamType], dt: Float)
    extends HasParams[ParamType]
    with TensorModule[ParamType] {
  private val modelVal = model
  private val dtVal = dt

  override def apply(input: Tensor[ParamType]): Tensor[ParamType] = {
    val gradMode = torch.enable_grad(true)
    try {
      var result = input
      for (layer <- modelVal.layers) {
        result = layer(result)
      }
      result
    } finally {
//      gradMode.close()
    }
  }

  def loss(x: Tensor[ParamType], t: Tensor[ParamType]): Tensor[ParamType] = {
    val gradMode = torch.enable_grad(true)
    try {
      var result = x
      for (layer <- modelVal.layers) {
        result = layer(result)
      }
      result
    } finally {
//      gradMode.close()
    }
  }

  def train(
      model: ODESolver[ParamType],
      dataset: Seq[(Tensor[ParamType], Tensor[ParamType])],
      optimizer: Optimizer,
      epochs: Int,
      batchSize: Int
  ): Unit = {
    model.train()
    var totalLoss = 0.0
    for (epoch <- 0 until epochs) {
      for (batch <- dataset) {
        val (inputs, labels) = batch
        optimizer.zero_grad()
        val outputs = model(inputs)
        val loss = model.loss(inputs, outputs)
        loss.backward()
        optimizer.step()
        totalLoss += loss.native.item_float()
      }
      println(s"Epoch ${epoch + 1}, Loss: ${totalLoss / dataset.length}")
    }
  }

}

// 定义 train 函数
//object TrainUtils[ParamType] {
//  def train(model: ODESolver, dataset: Seq[(Tensor[ParamType], Tensor[ParamType])], optimizer: Optimizer, epochs: Int, batchSize: Int): Unit = {
//    model.train()
//    var totalLoss = 0.0
//    for (epoch <- 0 until epochs) {
//      for (batch <- dataset) {
//        val (inputs, labels) = batch
//        optimizer.zero_grad()
//        val outputs = model.forward(inputs)
//        val loss = model.loss(inputs, outputs)
//        loss.backward()
//        optimizer.step()
//        totalLoss += loss.itemFloat()
//      }
//      println(s"Epoch ${epoch + 1}, Loss: ${totalLoss / dataset.length}")
//    }
//  }
//}
