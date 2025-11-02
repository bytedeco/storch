package torch
package distribute

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.chrono.Milliseconds
import org.bytedeco.pytorch
import org.bytedeco.pytorch.ReduceOp.RedOpType
import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.{
  AllToAllOptions,
  AllgatherOptions,
  AllreduceCoalescedOptions,
  AllreduceOptions,
  BarrierOptions,
  BroadcastOptions,
  DeviceOptional,
  DistributedBackend,
  DistributedBackendOptional,
  GatherOptions,
  LongVector,
  ProcessGroup,
  ProcessGroupCppCommHookInterface,
  ProcessGroupGloo,
  ReduceOptions,
  ReduceScatterOptions,
  ScatterOptions,
  TensorOptional,
  TensorVector,
  WorkInfoConsumer,
  gloo
}
import org.bytedeco.pytorch.global.torch.Backend

class ProcessGroupGlooSTorch[D <: DType](po: String) extends ProcessGroupGloo(new BytePointer(po)) {

  override def getBackendName: BytePointer = super.getBackendName

  override def getOptions: ProcessGroupGloo.Options = super.getOptions

  override def broadcast(tensors: TensorVector, opts: BroadcastOptions): pytorch.Work =
    super.broadcast(tensors, opts)

  override def broadcast(tensors: TensorVector): pytorch.Work = super.broadcast(tensors)

  override def allreduce(tensors: TensorVector, opts: AllreduceOptions): pytorch.Work =
    super.allreduce(tensors, opts)

  override def allreduce(tensors: TensorVector): pytorch.Work = super.allreduce(tensors)

  override def allreduce_sparse(tensors: TensorVector, opts: AllreduceOptions): pytorch.Work =
    super.allreduce_sparse(tensors, opts)

  override def allreduce_sparse(tensors: TensorVector): pytorch.Work =
    super.allreduce_sparse(tensors)

  override def allreduce_coalesced(
      tensors: TensorVector,
      opts: AllreduceCoalescedOptions
  ): pytorch.Work = super.allreduce_coalesced(tensors, opts)

  override def allreduce_coalesced(tensors: TensorVector): pytorch.Work =
    super.allreduce_coalesced(tensors)

  override def reduce(tensors: TensorVector, opts: ReduceOptions): pytorch.Work =
    super.reduce(tensors, opts)

  override def reduce(tensors: TensorVector): pytorch.Work = super.reduce(tensors)

  override def _reduce_scatter_base(
      outputTensor: pytorch.Tensor,
      inputTensor: pytorch.Tensor,
      opts: ReduceScatterOptions
  ): pytorch.Work = super._reduce_scatter_base(outputTensor, inputTensor, opts)

  override def _reduce_scatter_base(
      outputTensor: pytorch.Tensor,
      inputTensor: pytorch.Tensor
  ): pytorch.Work = super._reduce_scatter_base(outputTensor, inputTensor)

  override def _allgather_base(
      output_tensor: pytorch.Tensor,
      input_tensor: pytorch.Tensor,
      opts: AllgatherOptions
  ): pytorch.Work = super._allgather_base(output_tensor, input_tensor, opts)

  override def _allgather_base(
      output_tensor: pytorch.Tensor,
      input_tensor: pytorch.Tensor
  ): pytorch.Work = super._allgather_base(output_tensor, input_tensor)

  override def allgather(
      outputs: TensorVector,
      inputs: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather(outputs, inputs, opts)

  override def allgather(outputs: TensorVector, inputs: TensorVector): pytorch.Work =
    super.allgather(outputs, inputs)

  override def allgather_coalesced(
      output_lists: TensorVector,
      input_list: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather_coalesced(output_lists, input_list, opts)

  override def allgather_coalesced(
      output_lists: TensorVector,
      input_list: TensorVector
  ): pytorch.Work = super.allgather_coalesced(output_lists, input_list)

  override def allgather_into_tensor_coalesced(
      outputs: TensorVector,
      inputs: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather_into_tensor_coalesced(outputs, inputs, opts)

  override def allgather_into_tensor_coalesced(
      outputs: TensorVector,
      inputs: TensorVector
  ): pytorch.Work = super.allgather_into_tensor_coalesced(outputs, inputs)

  override def gather(
      outputs: TensorVector,
      inputs: TensorVector,
      opts: GatherOptions
  ): pytorch.Work = super.gather(outputs, inputs, opts)

  override def gather(outputs: TensorVector, inputs: TensorVector): pytorch.Work =
    super.gather(outputs, inputs)

  override def scatter(
      outputs: TensorVector,
      inputs: TensorVector,
      opts: ScatterOptions
  ): pytorch.Work = super.scatter(outputs, inputs, opts)

  override def scatter(outputs: TensorVector, inputs: TensorVector): pytorch.Work =
    super.scatter(outputs, inputs)

  override def reduce_scatter(
      outputs: TensorVector,
      inputs: TensorVector,
      opts: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter(outputs, inputs, opts)

  override def reduce_scatter(outputs: TensorVector, inputs: TensorVector): pytorch.Work =
    super.reduce_scatter(outputs, inputs)

  override def reduce_scatter_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts)

  override def reduce_scatter_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(outputTensors, inputTensors)

  override def alltoall_base(
      outputTensor: pytorch.Tensor,
      inputTensor: pytorch.Tensor,
      outputCounts: LongVector,
      inputCounts: LongVector,
      opts: AllToAllOptions
  ): pytorch.Work = super.alltoall_base(outputTensor, inputTensor, outputCounts, inputCounts, opts)

  override def alltoall_base(
      outputTensor: pytorch.Tensor,
      inputTensor: pytorch.Tensor,
      outputCounts: LongVector,
      inputCounts: LongVector
  ): pytorch.Work = super.alltoall_base(outputTensor, inputTensor, outputCounts, inputCounts)

  override def send(tensors: TensorVector, dstRank: Int, tag: Int): pytorch.Work =
    super.send(tensors, dstRank, tag)

  override def recv(tensors: TensorVector, srcRank: Int, tag: Int): pytorch.Work =
    super.recv(tensors, srcRank, tag)

  override def recvAnysource(tensors: TensorVector, tag: Int): pytorch.Work =
    super.recvAnysource(tensors, tag)

  override def barrier(opts: BarrierOptions): pytorch.Work = super.barrier(opts)

  override def barrier(): pytorch.Work = super.barrier()

  override def enableCollectivesTiming(): Unit = super.enableCollectivesTiming()

  override def _getStore(): gloo.Store = super._getStore()

  override def monitoredBarrier(opts: BarrierOptions, waitAllRanks: Boolean): Unit =
    super.monitoredBarrier(opts, waitAllRanks)

  override def monitoredBarrier(): Unit = super.monitoredBarrier()

  override def setSequenceNumberForGroup(): Unit = super.setSequenceNumberForGroup()

  override def getSequenceNumberForGroup: Long = super.getSequenceNumberForGroup

  override def getNumThreads: Int = super.getNumThreads
}
class ProcessGroupSTorch[D <: DType](po: String) extends ProcessGroup(new BytePointer(po)) {

  override def getRank: Int = super.getRank

  override def getSize: Int = super.getSize

  override def getID: Long = super.getID

  override def getBackendID(backend_type: ProcessGroup.BackendType): Long =
    super.getBackendID(backend_type)

  override def getBackendID(backend_type: Byte): Long = super.getBackendID(backend_type)

  override def getBackendName: BytePointer = super.getBackendName

  override def getBackendType: ProcessGroup.BackendType = super.getBackendType

  override def startCoalescing(deviceType: torch.DeviceType): Unit =
    super.startCoalescing(deviceType)

  override def startCoalescing(deviceType: Byte): Unit = super.startCoalescing(deviceType)

  override def endCoalescing(deviceType: torch.DeviceType): pytorch.Work =
    super.endCoalescing(deviceType)

  override def endCoalescing(deviceType: Byte): pytorch.Work = super.endCoalescing(deviceType)

  override def broadcast(tensors: TensorVector, opts: BroadcastOptions): pytorch.Work =
    super.broadcast(tensors, opts)

  override def broadcast(tensors: TensorVector): pytorch.Work = super.broadcast(tensors)

  override def allreduce(tensors: TensorVector, opts: AllreduceOptions): pytorch.Work =
    super.allreduce(tensors, opts)

  override def allreduce(tensors: TensorVector): pytorch.Work = super.allreduce(tensors)

  override def allreduce_coalesced(
      tensors: TensorVector,
      opts: AllreduceCoalescedOptions
  ): pytorch.Work = super.allreduce_coalesced(tensors, opts)

  override def allreduce_coalesced(tensors: TensorVector): pytorch.Work =
    super.allreduce_coalesced(tensors)

  override def reduce(tensors: TensorVector, opts: ReduceOptions): pytorch.Work =
    super.reduce(tensors, opts)

  override def reduce(tensors: TensorVector): pytorch.Work = super.reduce(tensors)

  override def allgather(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather(outputTensors, inputTensors, opts)

  override def allgather(outputTensors: TensorVector, inputTensors: TensorVector): pytorch.Work =
    super.allgather(outputTensors, inputTensors)

  override def _allgather_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor,
      opts: AllgatherOptions
  ): pytorch.Work = super._allgather_base(outputBuffer, inputBuffer, opts)

  override def _allgather_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor
  ): pytorch.Work = super._allgather_base(outputBuffer, inputBuffer)

  override def allgather_coalesced(
      outputTensorLists: TensorVector,
      inputTensors: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather_coalesced(outputTensorLists, inputTensors, opts)

  override def allgather_coalesced(
      outputTensorLists: TensorVector,
      inputTensors: TensorVector
  ): pytorch.Work = super.allgather_coalesced(outputTensorLists, inputTensors)

  override def allgather_into_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: AllgatherOptions
  ): pytorch.Work = super.allgather_into_tensor_coalesced(outputTensors, inputTensors, opts)

  override def allgather_into_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector
  ): pytorch.Work = super.allgather_into_tensor_coalesced(outputTensors, inputTensors)

  override def gather(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: GatherOptions
  ): pytorch.Work = super.gather(outputTensors, inputTensors, opts)

  override def gather(outputTensors: TensorVector, inputTensors: TensorVector): pytorch.Work =
    super.gather(outputTensors, inputTensors)

  override def scatter(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: ScatterOptions
  ): pytorch.Work = super.scatter(outputTensors, inputTensors, opts)

  override def scatter(outputTensors: TensorVector, inputTensors: TensorVector): pytorch.Work =
    super.scatter(outputTensors, inputTensors)

  override def reduce_scatter(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter(outputTensors, inputTensors, opts)

  override def reduce_scatter(
      outputTensors: TensorVector,
      inputTensors: TensorVector
  ): pytorch.Work = super.reduce_scatter(outputTensors, inputTensors)

  override def _reduce_scatter_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor,
      opts: ReduceScatterOptions
  ): pytorch.Work = super._reduce_scatter_base(outputBuffer, inputBuffer, opts)

  override def _reduce_scatter_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor
  ): pytorch.Work = super._reduce_scatter_base(outputBuffer, inputBuffer)

  override def reduce_scatter_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts)

  override def reduce_scatter_tensor_coalesced(
      outputTensors: TensorVector,
      inputTensors: TensorVector
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(outputTensors, inputTensors)

  override def alltoall_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor,
      outputSplitSizes: LongVector,
      inputSplitSizes: LongVector,
      opts: AllToAllOptions
  ): pytorch.Work =
    super.alltoall_base(outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts)

  override def alltoall_base(
      outputBuffer: pytorch.Tensor,
      inputBuffer: pytorch.Tensor,
      outputSplitSizes: LongVector,
      inputSplitSizes: LongVector
  ): pytorch.Work =
    super.alltoall_base(outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes)

  override def alltoall(
      outputTensors: TensorVector,
      inputTensors: TensorVector,
      opts: AllToAllOptions
  ): pytorch.Work = super.alltoall(outputTensors, inputTensors, opts)

  override def alltoall(outputTensors: TensorVector, inputTensors: TensorVector): pytorch.Work =
    super.alltoall(outputTensors, inputTensors)

  override def monitoredBarrier(opts: BarrierOptions, wait_all_ranks: Boolean): Unit =
    super.monitoredBarrier(opts, wait_all_ranks)

  override def monitoredBarrier(opts: BarrierOptions): Unit = super.monitoredBarrier(opts)

  override def setSequenceNumberForGroup(): Unit = super.setSequenceNumberForGroup()

  override def getSequenceNumberForGroup: Long = super.getSequenceNumberForGroup

  override def send(tensors: TensorVector, dstRank: Int, tag: Int): pytorch.Work =
    super.send(tensors, dstRank, tag)

  override def recv(tensors: TensorVector, srcRank: Int, tag: Int): pytorch.Work =
    super.recv(tensors, srcRank, tag)

  override def recvAnysource(tensors: TensorVector, tag: Int): pytorch.Work =
    super.recvAnysource(tensors, tag)

  override def barrier(opts: BarrierOptions): pytorch.Work = super.barrier(opts)

  override def barrier(): pytorch.Work = super.barrier()

//  override def getOptions: ProcessGroup.Options = super.getOptions

  override def hasBackends: Boolean = super.hasBackends

  override def setBackend(
      deviceType: torch.DeviceType,
      backendType: ProcessGroup.BackendType,
      backend: DistributedBackendOptional
  ): Unit = super.setBackend(deviceType, backendType, backend)

  override def setBackend(
      deviceType: Byte,
      backendType: Byte,
      backend: DistributedBackendOptional
  ): Unit = super.setBackend(deviceType, backendType, backend)

  override def getDefaultBackend: DistributedBackend = super.getDefaultBackend

  override def getBackend(deviceType: torch.DeviceType): DistributedBackend =
    super.getBackend(deviceType)

  override def getBackend(deviceType: Byte): DistributedBackend = super.getBackend(deviceType)

  override def getBackend(backendType: ProcessGroup.BackendType): DistributedBackend =
    super.getBackend(backendType)

  override def getDeviceTypes: pytorch.Device = super.getDeviceTypes

  override def registerOnCompletionHook(hook: WorkInfoConsumer): Unit =
    super.registerOnCompletionHook(hook)

  override def waitForPendingWorks(): Unit = super.waitForPendingWorks()

  override def hasHooks: Boolean = super.hasHooks

  override def getGroupName: BytePointer = super.getGroupName

  override def setGroupName(name: BytePointer): Unit = super.setGroupName(name)

  override def setGroupName(name: String): Unit = super.setGroupName(name)

  override def getGroupDesc: BytePointer = super.getGroupDesc

  override def setGroupDesc(name: BytePointer): Unit = super.setGroupDesc(name)

  override def setGroupDesc(name: String): Unit = super.setGroupDesc(name)

  override def enableCollectivesTiming(): Unit = super.enableCollectivesTiming()

  override def release_resources(): Unit = super.release_resources()

  override def getBoundDeviceId: DeviceOptional = super.getBoundDeviceId

  override def setBoundDeviceId(device: DeviceOptional): Unit = super.setBoundDeviceId(device)
}

enum ReduceOpType:
  case Sum, Avg, Product, Min, Max, Band, Bor, Bxor, PremulSum, Unused

  // 定义一个方法来获取对应的 RedOpType 实例
  def toNative: RedOpType = this match
    case Sum       => RedOpType.SUM
    case Avg       => RedOpType.AVG
    case Product   => RedOpType.PRODUCT
    case Min       => RedOpType.MIN
    case Max       => RedOpType.MAX
    case Band      => RedOpType.BAND
    case Bor       => RedOpType.BOR
    case Bxor      => RedOpType.BXOR
    case PremulSum => RedOpType.PREMUL_SUM
    case Unused    => RedOpType.UNUSED

//enum RedOpTypes(val value: Byte):
//  case SUM extends RedOpType(0.toByte)
//  case AVG extends RedOpType(1.toByte)
//  case PRODUCT extends RedOpType(2.toByte)
//  case MIN extends RedOpType(3.toByte)
//  case MAX extends RedOpType(4.toByte)
//  case BAND extends RedOpType(5.toByte) // Bitwise AND
//  case BOR extends RedOpType(6.toByte) // Bitwise OR
//  case BXOR extends RedOpType(7.toByte) // Bitwise XOR
//  case PREMUL_SUM extends RedOpType(8.toByte) // Multiply by a user-supplied constant before summing.
//  case UNUSED extends RedOpType(9.toByte)
//
//enum ReduceOpType(val value: Byte)  :
//  case SUM extends RedOpType.SUM //.(0.toByte)
//  case AVG extends RedOpType.AVG //(1.toByte)
//  case PRODUCT extends RedOpType.PRODUCT //(2.toByte)
//  case MIN extends RedOpType.MIN //(3.toByte)
//  case MAX extends RedOpType.MAX //(4.toByte)
//  case BAND extends RedOpType.BAND //(5.toByte) // Bitwise AND
//  case BOR extends RedOpType.BOR //(6.toByte) // Bitwise OR
//  case BXOR extends RedOpType.BXOR //(7.toByte) // Bitwise XOR
//  case PREMUL_SUM extends RedOpType.PREMUL_SUM //(8.toByte) // Multiply by a user-supplied constant before summing.
//  case UNUSED extends RedOpType.UNUSED //(9.toByte)

//https://pytorch.ac.cn/docs/stable/distributed.html#torch.distributed.TCPStore
class DistBackend[D <: DType](po: String) extends DistributedBackend(new BytePointer(po)) {

//  val native :Backend
  val native: DistributedBackend = new DistributedBackend(new BytePointer(po))

  override def getRank: Int = native.getRank

  override def getSize: Int = native.getSize

  override def getID: Long = native.getID

  override def supportsSplitting(): Boolean = native.supportsSplitting()

  override def startCoalescing(): Unit = native.startCoalescing()

  override def endCoalescing(): pytorch.Work = native.endCoalescing()

  override def getBackendName: BytePointer = native.getBackendName

  def getBackendNames: String = native.getBackendName.getString
//  override def broadcast(arg0: Seq[Tensor[D]], arg1: BroadcastOptions): pytorch.Work = native.broadcast(arg0, arg1)
//  override def broadcast(arg0: Seq[Tensor[D]]): pytorch.Work = super.broadcast(arg0)
//  override def allreduce(arg0: Seq[Tensor[D]], arg1: AllreduceOptions): pytorch.Work = super.allreduce(arg0, arg1)
//  override def allreduce(arg0: Seq[Tensor[D]]): pytorch.Work = super.allreduce(arg0)

  // torch.distributed.broadcast(tensor, src=None, group=None, async_op=False, group_src=None)[source]
  def broadcast(
      tensorSeq: Seq[Tensor[D]],
      async_op: Boolean = false,
      rootRank: Int,
      rootTensor: Int,
      timeout: Long,
      options: BroadcastOptions
  ): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    val option = new BroadcastOptions()
    option.timeout(new Milliseconds(timeout))
    option.asyncOp(async_op)
    option.rootRank(rootRank.toLong)
    option.rootTensor(rootTensor.toLong)
    native.broadcast(tensorVector, option)
  }

  def broadcast(tensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.broadcast(tensorVector)

  }
//torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
  // dist.all_reduce(tensor, async_op=True)。
  def allreduce(
      tensorSeq: Seq[Tensor[D]],
      tensorOpt: Option[Tensor[D]],
      reduceType: ReduceOpType = ReduceOpType.Sum,
      timeout: Long,
      options: AllreduceOptions
  ): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    val option = new AllreduceOptions()
    option.timeout(new Milliseconds(timeout))
    option.reduceOp(new org.bytedeco.pytorch.ReduceOp(reduceType.toNative))
    if tensorOpt.isDefined then option.sparseIndices().put(tensorOpt.get.native)
    native.allreduce(tensorVector, option)
  }

  def allreduce(tensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.allreduce(tensorVector)

  }

  def allreduce_sparse(
      tensorSeq: Seq[Tensor[D]],
      tensorOpt: Option[Tensor[D]],
      timeout: Long,
      reduceType: ReduceOpType = ReduceOpType.Sum,
      arg1: AllreduceOptions
  ): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    val option = new AllreduceOptions()
    if tensorOpt.isDefined then option.sparseIndices().put(tensorOpt.get.native)
    option.timeout(new Milliseconds(timeout))
    option.reduceOp(new org.bytedeco.pytorch.ReduceOp(reduceType.toNative))
    native.allreduce_sparse(tensorVector, option)
  }

  def allreduce_sparse(tensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.allreduce_sparse(tensorVector)

  }

  def allreduce_coalesced(
      tensorSeq: Seq[Tensor[D]],
      tensorOpt: Option[Tensor[D]],
      timeout: Long,
      arg1: AllreduceCoalescedOptions
  ): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    val option = new AllreduceCoalescedOptions()
    if tensorOpt.isDefined then option.sparseIndices().put(tensorOpt.get.native)
    option.timeout(new Milliseconds(timeout))
    native.allreduce_coalesced(tensorVector, option)
  }
  def allreduce_coalesced(tensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.allreduce_coalesced(tensorVector)
  }

  // torch.distributed.reduce(tensor, dst=None, op=<RedOpType.SUM: 0>, group=None, async_op=False, group_dst=None)[source]
  def reduce(
      tensorSeq: Seq[Tensor[D]],
      rootRank: Int,
      rootTensor: Int,
      timeout: Long,
      reduceType: ReduceOpType = ReduceOpType.Sum,
      arg1: ReduceOptions
  ): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    val option = new ReduceOptions()
    option.reduceOp(new org.bytedeco.pytorch.ReduceOp(reduceType.toNative))
    option.timeout(new Milliseconds(timeout))
    option.rootRank(rootRank.toLong)
    option.rootTensor(rootTensor)
    native.reduce(tensorVector, option)
  }
  def reduce(tensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.reduce(tensorVector)
  }

  // allgather(outputTensors: TensorVector, inputTensors: TensorVector,
  // torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
  def allgather(
      outputTensors: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]],
      timeout: Long,
      async_op: Boolean = false,
      arg2: AllgatherOptions
  ): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    val option = new AllgatherOptions()
    option.timeout(new Milliseconds(timeout))
    option.asyncOp(async_op)
    native.allgather(outputTensorVector, inputTensorVector, option)
  }
  // allgather(outputTensors: TensorVector, inputTensors: TensorVector, async_op: Boolean =false
  def allgather(
      outputTensors: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]],
      async_op: Boolean
  ): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    native.allgather(outputTensorVector, inputTensorVector)
  }
//allgather_coalesced(outputTensorLists: TensorVector, inputTensors: TensorVector
  def allgather_coalesced(
      outputTensorLists: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]],
      timeout: Long,
      async_op: Boolean = false,
      arg2: AllgatherOptions
  ): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensorLists.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    val option = new AllgatherOptions()
    option.timeout(new Milliseconds(timeout))
    option.asyncOp(async_op)
    native.allgather_coalesced(outputTensorVector, inputTensorVector, option)

  }
  // allgather_coalesced(outputTensorLists: TensorVector, inputTensors: TensorVector
  def allgather_coalesced(
      outputTensorLists: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]]
  ): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensorLists.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    native.allgather_coalesced(outputTensorVector, inputTensorVector)
  }

  // allgather_into_tensor_coalesced(outputTensors: TensorVector, inputTensors: TensorVector
  // torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
  def allgather_into_tensor_coalesced(
      outputTensors: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]],
      timeout: Long,
      async_op: Boolean = false,
      opts: AllgatherOptions
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    val option = new AllgatherOptions()
    option.timeout(new Milliseconds(timeout))
    option.asyncOp(async_op)
    native.allgather_into_tensor_coalesced(outTensorVector, inputTensorVector, option)
  }

  def allgather_into_tensor_coalesced(
      outputTensors: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]]
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    native.allgather_into_tensor_coalesced(outTensorVector, inputTensorVector)
  }
//gather(outputTensors: TensorVector, inputTensors: TensorVector, opts: GatherOptions)
  // torch.distributed.gather(tensor, gather_list=None, dst=None, group=None, async_op=False, group_dst=None)[source]
  def gather(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]],
      timeout: Long,
      rootRank: Int,
      opts: GatherOptions
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    val option = new GatherOptions()
    val ml = new Milliseconds()
    option.timeout(new Milliseconds(timeout))
    option.rootRank(rootRank)
    native.gather(outTensorVector, inputTensorVector, option)

  }

  def gather(outputTensorSeq: Seq[Tensor[D]], inputTensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    native.gather(outTensorVector, inputTensorVector)
  }

  // scatter(outputs: TensorVector, inputs: TensorVector, opts: ScatterOptions
  // torch.distributed.scatter(tensor, scatter_list=None, src=None, group=None, async_op=False, group_src=None)[source]
  def scatter(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]],
      async_op: Boolean = false,
      timeout: Long,
      rootRank: Int,
      opts: ScatterOptions
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    val option = new ScatterOptions()
    option.timeout(new Milliseconds(timeout))
    option.asyncOp(async_op)
    option.rootRank(rootRank.toLong)
    native.scatter(outTensorVector, inputTensorVector, option)
  }

  def scatter(outputTensorSeq: Seq[Tensor[D]], inputTensorSeq: Seq[Tensor[D]]): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    native.scatter(outTensorVector, inputTensorVector)
  }

  // torch.distributed.reduce_scatter(output, input_list, op=<RedOpType.SUM: 0>, group=None, async_op=False)
  def reduce_scatter(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]],
      async_op: Boolean = false,
      timeout: Long,
      reduceType: ReduceOpType = ReduceOpType.Sum,
      arg2: ReduceScatterOptions
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    val option = new ReduceScatterOptions()
    option.timeout(new Milliseconds(timeout))
    option.reduceOp(new org.bytedeco.pytorch.ReduceOp(reduceType.toNative))
    option.asyncOp(async_op)
    native.reduce_scatter(outTensorVector, inputTensorVector, option)

  }

  def reduce_scatter(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]]
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)

    native.reduce_scatter(outTensorVector, inputTensorVector)
  }

//torch.distributed.reduce_scatter_tensor(output, input, op=<RedOpType.SUM: 0>, group=None, async_op=False)[source]
  def reduce_scatter_tensor_coalesced(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]],
      async_op: Boolean = false,
      reduceType: ReduceOpType = ReduceOpType.Sum,
      timeout: Long,
      arg2: ReduceScatterOptions
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)
    val option = new ReduceScatterOptions()
    option.timeout(new Milliseconds(timeout))
    option.reduceOp(new org.bytedeco.pytorch.ReduceOp(reduceType.toNative))
    option.asyncOp(async_op)
    native.reduce_scatter_tensor_coalesced(outTensorVector, inputTensorVector, option)
  }

  def reduce_scatter_tensor_coalesced(
      outputTensorSeq: Seq[Tensor[D]],
      inputTensorSeq: Seq[Tensor[D]]
  ): pytorch.Work = {
    val outTensorVector: TensorVector = new TensorVector(outputTensorSeq.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensorSeq.map(_.native)*)

    native.reduce_scatter_tensor_coalesced(outTensorVector, inputTensorVector)
  }

  // alltoall_base(outputBuffer: pytorch.Tensor, inputBuffer: pytorch.Tensor, outputSplitSizes: LongVector, inputSplitSizes: LongVector, opts: AllToAllOptions): pytorch.Work = super.alltoall_base(outputBuffer, inputBuffer, outputSplitSizes, inputSplitSizes, opts)
  // torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)
  def alltoall_base(
      outputBuffer: Tensor[D],
      inputBuffer: Tensor[D],
      outputSplitSizes: Seq[Int],
      inputSplitSizes: Seq[Int],
      timeout: Long,
      opts: AllToAllOptions
  ): pytorch.Work = {

    val outSplitSeq = new LongVector(outputSplitSizes.map(_.toLong)*)
    val inputSplitSeq = new LongVector(inputSplitSizes.map(_.toLong)*)
    val option = new AllToAllOptions()
    option.timeout(new Milliseconds(timeout))
    native.alltoall_base(
      outputBuffer.native,
      inputBuffer.native,
      outSplitSeq,
      inputSplitSeq,
      option
    )

  }

  def alltoall_base(
      outputBuffer: Tensor[D],
      inputBuffer: Tensor[D],
      outputSplitSizes: Seq[Int],
      inputSplitSizes: Seq[Int]
  ): pytorch.Work = {
    val outSplitSeq = new LongVector(outputSplitSizes.map(_.toLong)*)
    val inputSplitSeq = new LongVector(inputSplitSizes.map(_.toLong)*)
    native.alltoall_base(outputBuffer.native, inputBuffer.native, outSplitSeq, inputSplitSeq)

  }

  // alltoall(outputTensors: TensorVector, inputTensors: TensorVector, opts: AllToAllOptions
  // torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
  def alltoall(
      outputTensors: Seq[Tensor[D]],
      inputTensors: Seq[Tensor[D]],
      timeout: Long,
      opts: AllToAllOptions
  ): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    val option = new AllToAllOptions()
    option.timeout(new Milliseconds(timeout))

    native.alltoall(outputTensorVector, inputTensorVector, option)
  }

  def alltoall(outputTensors: Seq[Tensor[D]], inputTensors: Seq[Tensor[D]]): pytorch.Work = {
    val outputTensorVector: TensorVector = new TensorVector(outputTensors.map(_.native)*)
    val inputTensorVector: TensorVector = new TensorVector(inputTensors.map(_.native)*)
    native.alltoall(outputTensorVector, inputTensorVector)
  }
//torch.distributed.isend(tensor, dst=None, group=None, tag=0, group_dst=None)[source]
  // torch.distributed.send(tensor, dst=None, group=None, tag=0, group_dst=None)
  // send(tensors: TensorVector, dstRank: Int, tag: Int
  def send(tensorSeq: Seq[Tensor[D]], dstRank: Int, tag: Int): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.send(tensorVector, dstRank, tag)
  }

//torch.distributed.recv(tensor, src=None, group=None, tag=0, group_src=None)
//recv(tensors: TensorVector, srcRank: Int, tag: Int
  def recv(tensorSeq: Seq[Tensor[D]], srcRank: Int, tag: Int): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.recv(tensorVector, srcRank, tag)
  }

//torch.distributed.recv_object_list(object_list, src=None, group=None, device=None, group_src=None)
  def recvAnysource(tensorSeq: Seq[Tensor[D]], tag: Int): pytorch.Work = {
    val tensorVector: TensorVector = new TensorVector(tensorSeq.map(_.native)*)
    native.recvAnysource(tensorVector, tag)
  }

  override def allreduce_sparse(arg0: TensorVector, arg1: AllreduceOptions): pytorch.Work =
    super.allreduce_sparse(arg0, arg1)

  override def allreduce_sparse(arg0: TensorVector): pytorch.Work = super.allreduce_sparse(arg0)

  override def allreduce_coalesced(
      arg0: TensorVector,
      arg1: AllreduceCoalescedOptions
  ): pytorch.Work = super.allreduce_coalesced(arg0, arg1)

  override def allreduce_coalesced(arg0: TensorVector): pytorch.Work =
    super.allreduce_coalesced(arg0)

  override def reduce(arg0: TensorVector, arg1: ReduceOptions): pytorch.Work =
    super.reduce(arg0, arg1)

  override def reduce(arg0: TensorVector): pytorch.Work = super.reduce(arg0)

  override def allgather(
      arg0: TensorVector,
      arg1: TensorVector,
      arg2: AllgatherOptions
  ): pytorch.Work = super.allgather(arg0, arg1, arg2)

  override def allgather(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.allgather(arg0, arg1)

  override def _allgather_base(
      arg0: pytorch.Tensor,
      arg1: pytorch.Tensor,
      arg2: AllgatherOptions
  ): pytorch.Work = super._allgather_base(arg0, arg1, arg2)

  override def _allgather_base(arg0: pytorch.Tensor, arg1: pytorch.Tensor): pytorch.Work =
    super._allgather_base(arg0, arg1)

  override def allgather_coalesced(
      arg0: TensorVector,
      arg1: TensorVector,
      arg2: AllgatherOptions
  ): pytorch.Work = super.allgather_coalesced(arg0, arg1, arg2)

  override def allgather_coalesced(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.allgather_coalesced(arg0, arg1)

  override def allgather_into_tensor_coalesced(
      arg0: TensorVector,
      arg1: TensorVector,
      arg2: AllgatherOptions
  ): pytorch.Work = super.allgather_into_tensor_coalesced(arg0, arg1, arg2)

  override def allgather_into_tensor_coalesced(
      arg0: TensorVector,
      arg1: TensorVector
  ): pytorch.Work = super.allgather_into_tensor_coalesced(arg0, arg1)

  override def gather(arg0: TensorVector, arg1: TensorVector, arg2: GatherOptions): pytorch.Work =
    super.gather(arg0, arg1, arg2)

  override def gather(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.gather(arg0, arg1)

  override def scatter(arg0: TensorVector, arg1: TensorVector, arg2: ScatterOptions): pytorch.Work =
    super.scatter(arg0, arg1, arg2)

  override def scatter(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.scatter(arg0, arg1)

  override def reduce_scatter(
      arg0: TensorVector,
      arg1: TensorVector,
      arg2: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter(arg0, arg1, arg2)

  override def reduce_scatter(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.reduce_scatter(arg0, arg1)

  override def _reduce_scatter_base(
      arg0: pytorch.Tensor,
      arg1: pytorch.Tensor,
      arg2: ReduceScatterOptions
  ): pytorch.Work = super._reduce_scatter_base(arg0, arg1, arg2)

  override def _reduce_scatter_base(arg0: pytorch.Tensor, arg1: pytorch.Tensor): pytorch.Work =
    super._reduce_scatter_base(arg0, arg1)

  override def reduce_scatter_tensor_coalesced(
      arg0: TensorVector,
      arg1: TensorVector,
      arg2: ReduceScatterOptions
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(arg0, arg1, arg2)

  override def reduce_scatter_tensor_coalesced(
      arg0: TensorVector,
      arg1: TensorVector
  ): pytorch.Work = super.reduce_scatter_tensor_coalesced(arg0, arg1)

  override def alltoall_base(
      arg0: pytorch.Tensor,
      arg1: pytorch.Tensor,
      arg2: LongVector,
      arg3: LongVector,
      arg4: AllToAllOptions
  ): pytorch.Work = super.alltoall_base(arg0, arg1, arg2, arg3, arg4)

  override def alltoall_base(
      arg0: pytorch.Tensor,
      arg1: pytorch.Tensor,
      arg2: LongVector,
      arg3: LongVector
  ): pytorch.Work = super.alltoall_base(arg0, arg1, arg2, arg3)

  override def alltoall(
      arg0: TensorVector,
      arg1: TensorVector,
      opts: AllToAllOptions
  ): pytorch.Work = super.alltoall(arg0, arg1, opts)

  override def alltoall(arg0: TensorVector, arg1: TensorVector): pytorch.Work =
    super.alltoall(arg0, arg1)
//torch.distributed.monitored_barrier(group=None, timeout=None, wait_all_ranks=False)
//torch.distributed.monitored_barrier(group=None, timeout=None, wait_all_ranks=False)
  override def monitoredBarrier(arg0: BarrierOptions, wait_all_ranks: Boolean): Unit =
    super.monitoredBarrier(arg0, wait_all_ranks)

  override def monitoredBarrier(arg0: BarrierOptions): Unit = super.monitoredBarrier(arg0)

  override def setSequenceNumberForGroup(): Unit = super.setSequenceNumberForGroup()

  override def getSequenceNumberForGroup: Long = super.getSequenceNumberForGroup

  override def send(arg0: TensorVector, arg1: Int, arg2: Int): pytorch.Work =
    super.send(arg0, arg1, arg2)

  override def recv(arg0: TensorVector, arg1: Int, arg2: Int): pytorch.Work =
    super.recv(arg0, arg1, arg2)

  override def recvAnysource(arg0: TensorVector, arg1: Int): pytorch.Work =
    super.recvAnysource(arg0, arg1)

//torch.distributed.barrier(group=None, async_op=False, device_ids=None)
  override def barrier(arg0: BarrierOptions): pytorch.Work = super.barrier(arg0)

  override def barrier(): pytorch.Work = super.barrier()

  override def registerOnCompletionHook(hook: WorkInfoConsumer): Unit =
    super.registerOnCompletionHook(hook)

  override def waitForPendingWorks(): Unit = super.waitForPendingWorks()

  override def enableCollectivesTiming(): Unit = super.enableCollectivesTiming()

  override def hasHooks: Boolean = super.hasHooks

  override def setGroupUid(pg_uid: BytePointer): Unit = super.setGroupUid(pg_uid)

  override def setGroupUid(pg_uid: String): Unit = super.setGroupUid(pg_uid)

  override def getGroupUid: BytePointer = super.getGroupUid

  override def setGroupDesc(desc: BytePointer): Unit = super.setGroupDesc(desc)

  override def setGroupDesc(desc: String): Unit = super.setGroupDesc(desc)

  override def getGroupDesc: BytePointer = super.getGroupDesc

  override def getBoundDeviceId: DeviceOptional = super.getBoundDeviceId

  override def eagerConnectSingleDevice(device: pytorch.Device): Unit =
    super.eagerConnectSingleDevice(device)

  override def setBoundDeviceId(device: DeviceOptional): Unit = super.setBoundDeviceId(device)
}
