/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn
package modules

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InputArchive, OutputArchive}
import torch.internal.NativeConverters.fromNative

import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}

abstract class Module {

  protected[torch] var _nativeModule = pytorch.Module()
  private[torch] def nativeModule: pytorch.Module = _nativeModule // = pytorch.Module()
  private var childModules: TreeSeqMap[String, Module] = TreeSeqMap.empty

  def named_buffers(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val buffers = nativeModule.named_buffers(recurse)
    TreeSeqMap.from((0 until buffers.size().toInt).map { i =>
      val item = buffers.get(i)
      (item.key().getString(), fromNative[DType](item.access()))
    })

  def named_parameters(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val params = nativeModule.named_parameters(recurse)
    TreeSeqMap.from((0 until params.size().toInt).map { i =>
      val item = params.get(i)
      (item.key().getString(), fromNative[DType](item.access()))
    })
  def namedBuffers(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val buffers = nativeModule.named_buffers(recurse)
    TreeSeqMap.from((0 until buffers.size().toInt).map { i =>
      val item = buffers.get(i)
      (item.key().getString(), fromNative[DType](item.access()))
    })

  def namedParameters(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val params = nativeModule.named_parameters(recurse)
    TreeSeqMap.from((0 until params.size().toInt).map { i =>
      val item = params.get(i)
      (item.key().getString(), fromNative[DType](item.access()))
    })

  def parameters(un_used: Int*): Seq[Tensor[?]] = parameters(recurse = true)

  def parameters: Seq[Tensor[?]] = parameters(recurse = true)

  def parameters(recurse: Boolean): Seq[Tensor[?]] =
    ArraySeq.unsafeWrapArray(nativeModule.parameters().get).map(fromNative[DType])

  def stateDict(): Map[String, Tensor[?]] = {
    val tensorsToLoad = namedParameters() ++ namedBuffers()
    tensorsToLoad
  }
  def state_dict() = stateDict()

  // TODO make strict a parameter
  // TODO improve error handling

  def load_state_dict(stateDict: Map[String, Tensor[DType]]): Unit =
    val tensorsToLoad = named_parameters() ++ named_buffers()
    // assert(stateDict.keySet -- tensorsToLoad.keySet == Set.empty, s"keys missing in state dict: ${tensorsToLoad.keySet -- stateDict.keySet}")
    for ((key, param) <- tensorsToLoad if stateDict.contains(key))
      noGrad {
        param.copy_(stateDict(key))
      }

  def loadStateDict(stateDict: Map[String, Tensor[DType]]): Unit =
    val tensorsToLoad = named_parameters() ++ named_buffers()
    // assert(stateDict.keySet -- tensorsToLoad.keySet == Set.empty, s"keys missing in state dict: ${tensorsToLoad.keySet -- stateDict.keySet}")
    for ((key, param) <- tensorsToLoad if stateDict.contains(key))
      noGrad {
        param.copy_(stateDict(key))
      }

  def modules(recurse: Boolean): Seq[Module] =
    childModules.values.flatMap(child => child +: child.modules).toSeq.distinct

  def modules: Seq[Module] = modules(recurse = true)

  def namedChildren: SeqMap[String, Module] = childModules

  def named_children: SeqMap[String, Module] = childModules

  def children(): Seq[Module] = modules(true)

  def named_modules: SeqMap[String, Module] =
    namedChildren.flatMap((_, module) => module.namedModules)

  def namedModules: SeqMap[String, Module] =
    namedChildren.flatMap((_, module) => module.namedModules)

  def apply(fn: Module => Unit): this.type =
    for (_, module) <- namedModules
    do module(fn)
    this

  def register[M <: Module](child: M, n: String = "")(using name: sourcecode.Name): M =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()

    childModules = childModules.updated(name_, child)
    nativeModule.register_module(name_, child.nativeModule)
    child

  def registerModule[M <: Module](child: M, n: String = "")(using name: sourcecode.Name): M =
    register(child = child)(using name)

  def register_module[M <: Module](child: M, n: String = "")(using name: sourcecode.Name): M =
    register(child = child)(using name)

  def register_module[M <: Module](name: String, child: M): M =
    register(child = child)(using name)

  def unregister_module[M <: Module](name: String): Unit =
    unregister_module(name)

  def registerParameter[D <: DType](t: Tensor[D], requiresGrad: Boolean = true, n: String = "")(
      using name: sourcecode.Name
  ): Tensor[D] =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    nativeModule.register_parameter(name_, t.native, requiresGrad)
    t

  def register_parameter_r[D <: DType](t: Tensor[D], requires_grad: Boolean = true, n: String = "")(
      using name: sourcecode.Name
  ): Tensor[D] =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    nativeModule.register_parameter(name_, t.native, requires_grad)
    t

  def register_parameter[D <: DType](
      name: String,
      t: Tensor[D],
      requires_grad: Boolean = true
  ): Tensor[D] =
    nativeModule.register_parameter(name, t.native, requires_grad)
    t

  def registerBuffer[D <: DType](t: Tensor[D], n: String = "")(using
      name: sourcecode.Name
  ): Tensor[D] =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    nativeModule.register_buffer(name_, t.native)
    t

  /** Adds a buffer to the module. */
  def registerBuffer[D <: DType](name: String, tensor: Tensor[D]): Tensor[D] =
    fromNative(nativeModule.register_buffer(name, tensor.native))

  def register_buffer[D <: DType](
      name: String,
      tensor: Tensor[D],
      persistent: Boolean = false
  ): Tensor[D] =
    fromNative(nativeModule.register_buffer(name, tensor.native))

  def hasBias(): Boolean = modules.exists(_.hasBias())

  def has_bias(): Boolean = modules.exists(_.hasBias())

  def eval(): Unit = nativeModule.eval()

  def isTraining: Boolean = nativeModule.is_training

  def is_training: Boolean = nativeModule.is_training

  def is_training(un_used: Int*): Boolean = nativeModule.is_training

  def train(on: Boolean = true): Unit = nativeModule.train(on)

  def is_serializable: Boolean = nativeModule.is_serializable

  def is_serializable(un_used: Int*): Boolean = nativeModule.is_serializable

  def to(device: Device): this.type =
    nativeModule.to(device.toNative, false)
    this

  def to(device: Device, non_blocking: Boolean = false): this.type =
    nativeModule.to(device.toNative, non_blocking)
    this

  def zero_grad(set_to_none: Boolean = false): Unit = nativeModule.zero_grad(set_to_none)

  def save(outputArchive: OutputArchive) = nativeModule.save(outputArchive)

  def load(inputArchive: InputArchive) = nativeModule.load(inputArchive)

  def asTransformer = nativeModule.asTransformer()

  def asTransformerEncoder = nativeModule.asTransformerEncoder()

  def asTransformerDecoder = nativeModule.asTransformerDecoder()

  def asMultiheadAttention = nativeModule.asMultiheadAttention()

  def asTransformerEncoderLayer = nativeModule.asTransformerEncoderLayer()

  def asTransformerDecoderLayer = nativeModule.asTransformerDecoderLayer()

  def asGroupNorm = nativeModule.asGroupNorm()

  def asCrossMapLRN2d = nativeModule.asCrossMapLRN2d()

  def asLocalResponseNorm = nativeModule.asLocalResponseNorm()

  def asLayerNorm = nativeModule.asLayerNorm()

  def asUpsample = nativeModule.asUpsample()

  def asPixelUnshuffle = nativeModule.asPixelUnshuffle()

  def asPixelShuffle = nativeModule.asPixelShuffle()

  def asGRUCell = nativeModule.asGRUCell()

  def asLSTMCell = nativeModule.asLSTMCell()

  def asRNNCell = nativeModule.asRNNCell()

  def asGRU = nativeModule.asGRU()

  def asLSTM = nativeModule.asLSTM()

  def asRNN = nativeModule.asRNN()

  def asLPPool3d = nativeModule.asLPPool3d()

  def asFractionalMaxPool3d = nativeModule.asFractionalMaxPool3d()

  def asMaxUnpool3d = nativeModule.asMaxUnpool3d()

  def asAdaptiveMaxPool3d = nativeModule.asAdaptiveMaxPool3d()

  def asAdaptiveAvgPool3d = nativeModule.asAdaptiveAvgPool3d()

  def asMaxPool3d = nativeModule.asMaxPool3d()

  def asAvgPool3d = nativeModule.asAvgPool3d()

  def asZeroPad3d = nativeModule.asZeroPad3d()

  def asConstantPad3d = nativeModule.asConstantPad3d()

  def asReplicationPad3d = nativeModule.asReplicationPad3d()

  def asReflectionPad3d = nativeModule.asReflectionPad3d()

  def asLPPool2d = nativeModule.asLPPool2d()

  def asFractionalMaxPool2d = nativeModule.asFractionalMaxPool2d()

  def asMaxUnpool2d = nativeModule.asMaxUnpool2d()

  def asAdaptiveMaxPool2d = nativeModule.asAdaptiveMaxPool2d()

  def asAdaptiveAvgPool2d = nativeModule.asAdaptiveAvgPool2d()

  def asMaxPool2d = nativeModule.asMaxPool2d()

  def asAvgPool2d = nativeModule.asAvgPool2d()

  def asZeroPad2d = nativeModule.asZeroPad2d()

  def asConstantPad2d = nativeModule.asConstantPad2d()

  def asReplicationPad2d = nativeModule.asReplicationPad2d()

  def asReflectionPad2d = nativeModule.asReflectionPad2d()

  def asLPPool1d = nativeModule.asLPPool1d()

  def asMaxUnpool1d = nativeModule.asMaxUnpool1d()

  def asAdaptiveMaxPool1d = nativeModule.asAdaptiveMaxPool1d()

  def asAdaptiveAvgPool1d = nativeModule.asAdaptiveAvgPool1d()

  def asMaxPool1d = nativeModule.asMaxPool1d()

  def asAvgPool1d = nativeModule.asAvgPool1d()

  def asZeroPad1d = nativeModule.asZeroPad1d()

  def asConstantPad1d = nativeModule.asConstantPad1d()

  def asReplicationPad1d = nativeModule.asReplicationPad1d()

  def asReflectionPad1d = nativeModule.asReflectionPad1d()

  def asUnflatten = nativeModule.asUnflatten()

  def asFlatten = nativeModule.asFlatten()

  def asBilinear = nativeModule.asBilinear()

  def asLinear = nativeModule.asLinear()

  def asIdentity = nativeModule.asIdentity()

  def asUnfold = nativeModule.asUnfold()

  def asFold = nativeModule.asFold()

  def asEmbeddingBag = nativeModule.asEmbeddingBag()

  def asEmbedding = nativeModule.asEmbedding()

  def asPairwiseDistance = nativeModule.asPairwiseDistance()

  def asCosineSimilarity = nativeModule.asCosineSimilarity()

  def asFeatureAlphaDropout = nativeModule.asFeatureAlphaDropout()

  def asAlphaDropout = nativeModule.asAlphaDropout()

  def asDropout3d = nativeModule.asDropout3d()

  def asConvTranspose3d = nativeModule.asConvTranspose3d()

  def asConv3d = nativeModule.asConv3d()

  def asInstanceNorm3d = nativeModule.asInstanceNorm3d()

  def asBatchNorm3d = nativeModule.asBatchNorm3d()

  def asDropout2d = nativeModule.asDropout2d()

  def asConvTranspose2d = nativeModule.asConvTranspose2d()

  def asConv2d = nativeModule.asConv2d()

  def asInstanceNorm2d = nativeModule.asInstanceNorm2d()

  def asBatchNorm2d = nativeModule.asBatchNorm2d()

  def asDropout = nativeModule.asDropout()

  def asConvTranspose1d = nativeModule.asConvTranspose1d()

  def asConv1d = nativeModule.asConv1d()

  def asAdaptiveLogSoftmaxWithLoss = nativeModule.asAdaptiveLogSoftmaxWithLoss()

  def asParameterList = nativeModule.asParameterList()

  def asParameterDict = nativeModule.asParameterDict()

  def asSequential = nativeModule.asSequential()

  def asModuleList = nativeModule.asModuleList()

  def asModuleDict = nativeModule.asModuleDict()

  def asL1Loss = nativeModule.asL1Loss()

  def asKLDivLoss = nativeModule.asKLDivLoss()

  def asMSELoss = nativeModule.asMSELoss()

  def asBCELoss = nativeModule.asBCELoss()

  def asHingeEmbeddingLoss = nativeModule.asHingeEmbeddingLoss()

  def asMultiMarginLoss = nativeModule.asMultiMarginLoss()

  def asCosineEmbeddingLoss = nativeModule.asCosineEmbeddingLoss()

  def asSmoothL1Loss = nativeModule.asSmoothL1Loss()

  def asHuberLoss = nativeModule.asHuberLoss()

  def asMultiLabelMarginLoss = nativeModule.asMultiLabelMarginLoss()

  def asSoftMarginLoss = nativeModule.asSoftMarginLoss()

  def asMultiLabelSoftMarginLoss = nativeModule.asMultiLabelSoftMarginLoss()

  def asTripletMarginLoss = nativeModule.asTripletMarginLoss()

  def asTripletMarginWithDistanceLoss = nativeModule.asTripletMarginWithDistanceLoss()

  def asCTCLoss = nativeModule.asCTCLoss()

  def asPoissonNLLLoss = nativeModule.asPoissonNLLLoss()

  def asMarginRankingLoss = nativeModule.asMarginRankingLoss()

  def asNLLLoss = nativeModule.asNLLLoss()

  def asCrossEntropyLoss = nativeModule.asCrossEntropyLoss()

  def asBCEWithLogitsLoss = nativeModule.asBCEWithLogitsLoss()

  override def toString(): String = {
    val className = getClass().getName()
    className.split("\\$").last
    // getClass().getSimpleName()
  }

  private def doSummarize(indent: Int): String =
    val thisModule = toString
    if modules.isEmpty then thisModule
    else
      thisModule + namedChildren
        .map((name, module) => s"${" " * (indent + 2)}($name): " + module.doSummarize(indent + 2))
        .mkString("(\n", "\n", s"\n${" " * indent})")
  def summarize: String =
    doSummarize(0)
}

trait HasParams[ParamType <: FloatNN | ComplexNN: Default] extends Module:
  override def parameters(recurse: Boolean): Seq[Tensor[ParamType]] =
    nativeModule.parameters(recurse).get().toSeq.map(fromNative[ParamType])
  override def parameters: Seq[Tensor[ParamType]] = parameters(recurse = true)
  transparent inline def paramType: DType = summon[Default[ParamType]].dtype
//  def forward(input: Tensor[ParamType]): Tensor[ParamType]

trait HasWeight[ParamType <: FloatNN | ComplexNN]:
  def weight: Tensor[ParamType]

/** Transforms a single tensor into another one of the same type. */
trait TensorModule[D <: DType] extends Module with (Tensor[D] => Tensor[D]):
  override def toString(): String = "TensorModule"

trait TensorModuleBase[D <: DType, D2 <: DType] extends Module with (Tensor[D] => Tensor[D2]) {
  override def toString() = "TensorModuleBase"
}
