package torch.utils.models.education.deepirt.layer

import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.{Module, Parameter, functional as F}
import torch.{Default, FloatNN, Tensor, nn}

import scala.collection.mutable.ArrayBuffer

class DeepIRT[ParamType <: FloatNN: Default](
    val memorySize: Int,
    val memoryKeyStateDim: Int,
    val memoryValueStateDim: Int,
    initMemoryKey: Tensor[ParamType]
) extends Module
    with HasParams[ParamType] {

  val keyHead = DeepIRTLayer[ParamType](memorySize, memoryKeyStateDim, isWrite = false)
  val valueHead = DeepIRTLayer[ParamType](memorySize, memoryValueStateDim, isWrite = true)

  var memoryKey: Tensor[ParamType] = initMemoryKey
  var memoryValue: Option[Tensor[ParamType]] = None

  def init_value_memory(memoryValue: Tensor[ParamType]): Unit = {
    this.memoryValue = Some(memoryValue)
  }

  def attention(controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    keyHead.addressing(controlInput, memoryKey)
  }

  def read(readWeight: Tensor[ParamType]): Tensor[ParamType] = {
    require(memoryValue.isDefined, "memoryValue not initialized. Call initValueMemory first.")
    valueHead.read(memoryValue.get, readWeight = Some(readWeight))
  }

  def write(writeWeight: Tensor[ParamType], controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    require(memoryValue.isDefined, "memoryValue not initialized. Call initValueMemory first.")
    val newMemoryValue = valueHead.write(controlInput, memoryValue.get, writeWeight)
    memoryValue = Some(newMemoryValue)
    newMemoryValue
  }

  override def parameters: List[Tensor[ParamType]] = {
    keyHead.parameters ++ valueHead.parameters
  }

  def apply(controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    val attnWeight = attention(controlInput)
    read(attnWeight)
  }
}
