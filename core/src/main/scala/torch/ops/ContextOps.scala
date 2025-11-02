package torch
package ops

import org.bytedeco.pytorch.global.{torch as torchNative}
import org.bytedeco.pytorch.{
  DoubleOptional,
  LongOptional,
  PackedSequence,
  Scalar,
  ScalarOptional,
  TensorOptional,
  TensorOptionalList,
  TensorVector,
  Context,
  LongVector
}
import torch.internal.NativeConverters.{fromNative, toScalar}

import scala.collection.mutable.ListBuffer

private[torch] trait ContextOps {

  def globalContext() = torchNative.globalContext()

  given context: Context = torchNative.globalContext() // new Context()

  def setQEngine(engine: torchNative.QEngine)(using
      context: Context = torchNative.globalContext()
  ) = context.setQEngine(engine)

  def setSDPPriorityOrder(order: Array[Long])(using
      context: Context = torchNative.globalContext()
  ): Unit =
    context.setSDPPriorityOrder(new LongVector(order*))

  def setBenchmarkLimitCuDNN(limit: Int)(using context: Context = torchNative.globalContext()) =
    context.setBenchmarkLimitCuDNN(limit)

  def hasOpenMP = Context.hasOpenMP()

  def hasMKL = Context.hasMKL()

  def hasKleidiAI = Context.hasKleidiAI()

  def hasLAPACK = Context.hasLAPACK()

  def hasMKLDNN = Context.hasMKLDNN()

  def hasMAGMA = Context.hasMAGMA()

  def hasCUDA = Context.hasCUDA()

  def hasMTIA = Context.hasMTIA()

  def hasCUDART = Context.hasCUDART()

  def hasCuDNN = Context.hasCuDNN()

  def hasCuSOLVER = Context.hasCuSOLVER()

  def hasCuBLASLt = Context.hasCuBLASLt()

  def hasROCM = Context.hasROCM()

  def hasHIP = Context.hasHIP()

  def hasMPS = Context.hasMPS()

  def hasIPU = Context.hasIPU()

  def hasXLA = Context.hasXLA()

  def hasXPU = Context.hasXPU()

  def hasLazy = Context.hasLazy()

  def hasMAIA = Context.hasMAIA()

  def hasHPU = Context.hasHPU()

  def versionCuDNN: Long = Context.versionCuDNN()

  def versionCUDART: Long = Context.versionCUDART()

  def getNVRTC = Context.getNVRTC()

  def lazyInitCUDA(using context: Context = torchNative.globalContext()) = context.lazyInitCUDA()

  def lazyInitHIP(using context: Context = torchNative.globalContext()) = context.lazyInitHIP()

  def lazyInitXPU(using context: Context = torchNative.globalContext()) = context.lazyInitXPU()

  def lazyInitMTIA(using context: Context = torchNative.globalContext()) = context.lazyInitMTIA()

  def lazyInitPrivateUse1(using context: Context = torchNative.globalContext()) =
    context.lazyInitPrivateUse1()

  def unsetDefaultMobileCPUAllocator(using context: Context = torchNative.globalContext()) =
    context.unsetDefaultMobileCPUAllocator()

  def setDefaultMobileCPUAllocator(using context: Context = torchNative.globalContext()) =
    context.setDefaultMobileCPUAllocator()

  def qEngine(using context: Context) = context.qEngine()

  def supportedQEngines = Context.supportedQEngines()

  def alertCuBLASConfigNotDeterministic(using context: Context = torchNative.globalContext()) =
    context.alertCuBLASConfigNotDeterministic()

  def setFloat32MatmulPrecision(precision: String)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setFloat32MatmulPrecision(precision)

  def alertNotDeterministic(caller: String) = Context.alertNotDeterministic(caller)

  def deterministicFillUninitializedMemory(using context: Context = torchNative.globalContext()) =
    context.deterministicFillUninitializedMemory()

  def deterministicAlgorithms(using context: Context = torchNative.globalContext()) =
    context.deterministicAlgorithms()

  def deterministicAlgorithmsWarnOnly(using context: Context = torchNative.globalContext()) =
    context.deterministicAlgorithmsWarnOnly()

  def float32MatmulPrecision(using context: Context = torchNative.globalContext()) =
    context.float32MatmulPrecision()

  def getROCmFAPreferredBackend(using context: Context = torchNative.globalContext()) =
    context.getROCmFAPreferredBackend()

  def blasPreferredBackend(using context: Context = torchNative.globalContext()) =
    context.blasPreferredBackend()

  def userEnabledNNPACK(using context: Context = torchNative.globalContext()) =
    context.userEnabledNNPACK()

  def sDPPriorityOrder(using context: Context = torchNative.globalContext()) =
    context.sDPPriorityOrder()

//  def userEnabledNNPACK(using context: Context): Boolean = context.userEnabledNNPACK()

  def deterministicCuDNN(using context: Context = torchNative.globalContext()) =
    context.deterministicCuDNN()

  def deterministicMkldnn(using context: Context = torchNative.globalContext()): Boolean =
    context.deterministicMkldnn()

  def benchmarkLimitCuDNN(using context: Context = torchNative.globalContext()) =
    context.benchmarkLimitCuDNN()

  def benchmarkCuDNN(using context: Context = torchNative.globalContext()) =
    context.benchmarkCuDNN()

  def setBenchmarkCuDNN(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setBenchmarkCuDNN(enabled)

  def setDeterministicCuDNN(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setDeterministicCuDNN(enabled)

  def setDeterministicMkldnn(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setDeterministicMkldnn(enabled)

  def setUserEnabledNNPACK(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setUserEnabledNNPACK(enabled)

  def setUserEnabledCuDNN(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setUserEnabledCuDNN(enabled)

  def setUserEnabledMkldnn(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setUserEnabledMkldnn(enabled)

  def setSDPUseFlash(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setSDPUseFlash(enabled)

  def setSDPUseMemEfficient(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setSDPUseMemEfficient(enabled)

  def setSDPUseMath(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setSDPUseMath(enabled)

  def setSDPUseCuDNN(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setSDPUseCuDNN(enabled)

  def userEnabledCuDNN(using context: Context) = context.userEnabledCuDNN()

  def userEnabledMkldnn(using context: Context) = context.userEnabledMkldnn()

  def userEnabledFlashSDP(using context: Context = torchNative.globalContext()) =
    context.userEnabledFlashSDP()

  def userEnabledMemEfficientSDP(using context: Context = torchNative.globalContext()) =
    context.userEnabledMemEfficientSDP()

  def userEnabledMathSDP(using context: Context = torchNative.globalContext()) =
    context.userEnabledMathSDP()

  def userEnabledCuDNNSDP(using context: Context = torchNative.globalContext()) =
    context.userEnabledCuDNNSDP()

  def setAllowFP16BF16ReductionMathSDP(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setAllowFP16BF16ReductionMathSDP(enabled)

  def setSDPUseOverrideable(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setSDPUseOverrideable(enabled)

  def setDeterministicFillUninitializedMemory(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setDeterministicFillUninitializedMemory(enabled)

  def setDeterministicAlgorithms(enabled: Boolean, en: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setDeterministicAlgorithms(enabled, en)

  def setAllowTF32CuDNN(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setAllowTF32CuDNN(enabled)

  def setAllowTF32OneDNN(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setAllowTF32OneDNN(enabled)

  def setAllowTF32CuBLAS(enabled: Boolean)(using context: Context = torchNative.globalContext()) =
    context.setAllowTF32CuBLAS(enabled)

  def setAllowFP16ReductionCuBLAS(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setAllowFP16ReductionCuBLAS(enabled)

  def setAllowBF16ReductionCuBLAS(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setAllowBF16ReductionCuBLAS(enabled)

  def setAllowFP16AccumulationCuBLAS(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setAllowFP16AccumulationCuBLAS(enabled)

  def setCheckSparseTensorInvariants(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setCheckSparseTensorInvariants(enabled)

  def setReleaseWeightsWhenPrepacking(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setReleaseWeightsWhenPrepacking(enabled)

  def setAllowFP16ReductionCPU(enabled: Boolean)(using
      context: Context = torchNative.globalContext()
  ) =
    context.setAllowFP16ReductionCPU(enabled)

  def allowFP16BF16ReductionMathSDP(using context: Context = torchNative.globalContext()) =
    context.allowFP16BF16ReductionMathSDP()

  def userEnabledOverrideableSDP(using context: Context = torchNative.globalContext()) =
    context.userEnabledOverrideableSDP()

  def allowTF32CuDNN(using context: Context = torchNative.globalContext()) =
    context.allowTF32CuDNN()

  def allowTF32OneDNN(using context: Context = torchNative.globalContext()) =
    context.allowTF32OneDNN()

  def allowTF32CuBLAS(using context: Context = torchNative.globalContext()) =
    context.allowTF32CuBLAS()

  def allowFP16ReductionCuBLAS(using context: Context = torchNative.globalContext()) =
    context.allowFP16ReductionCuBLAS()

  def allowBF16ReductionCuBLAS(using context: Context) = context.allowBF16ReductionCuBLAS()

  def isXNNPACKAvailable = Context.isXNNPACKAvailable()

  def checkSparseTensorInvariants(using context: Context) = context.checkSparseTensorInvariants()

  def releaseWeightsWhenPrepacking(using context: Context) = context.releaseWeightsWhenPrepacking()

  def areVmapFallbackWarningsEnabled(using context: Context = torchNative.globalContext()) =
    context.areVmapFallbackWarningsEnabled()

  def isDefaultMobileCPUAllocatorSet(using context: Context = torchNative.globalContext()) =
    context.isDefaultMobileCPUAllocatorSet()

  def allowFP16ReductionCPU(using context: Context = torchNative.globalContext()) =
    context.allowFP16ReductionCPU()

  def allowFP16AccumulationCuBLAS(using context: Context = torchNative.globalContext()) =
    context.allowFP16AccumulationCuBLAS()

}
