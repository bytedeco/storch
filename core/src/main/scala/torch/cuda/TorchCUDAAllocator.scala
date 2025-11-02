package torch.cuda

import org.bytedeco.cuda.cudart.CUstream_st
import org.bytedeco.javacpp.{BytePointer, Pointer, SizeTPointer}
import org.bytedeco.pytorch.cuda.*
import org.bytedeco.pytorch.global.torch_cuda
import org.bytedeco.pytorch.{DataPtr, StringPair}

import java.io.Closeable

/** Scala 3 包装器：封装 JavaCPP 的 CUDAAllocator，提供类型安全的 CUDA 内存分配器接口 用于管理 CUDA 设备上的内存分配和跟踪内存使用情况
  */
class TorchCUDAAllocator(val nativeAllocator: CUDAAllocator) extends Closeable {

  def raw_alloc(nbytes: Long) = nativeAllocator.raw_alloc(nbytes)

  def raw_alloc_with_stream(nbytes: Long, stream: CUstream_st) =
    nativeAllocator.raw_alloc_with_stream(nbytes, stream)

  def raw_delete(ptr: Pointer) = nativeAllocator.raw_delete(ptr)

  def init(device_count: Int) = nativeAllocator.init(device_count)

  def initialized(): Boolean = nativeAllocator.initialized()

  def getMemoryFraction(device: Byte) = nativeAllocator.getMemoryFraction(device)

  def setMemoryFraction(fraction: Double, device: Byte) =
    nativeAllocator.setMemoryFraction(fraction, device)

  def emptyCache = nativeAllocator.emptyCache()

  def enable(value: Boolean) = nativeAllocator.enable(value)

  def isEnabled = nativeAllocator.isEnabled()

  def getBaseAllocation(ptr: Pointer, size: SizeTPointer) =
    nativeAllocator.getBaseAllocation(ptr, size)
  def recordStream(arg: DataPtr, stream: CUDAStream) = nativeAllocator.recordStream(arg, stream)

  /** 获取指定设备的 CUDA 内存分配器统计信息
    * @param device
    *   设备索引
    * @return
    *   包含内存使用统计信息的 DeviceStats 对象
    */
  def getDeviceStats(device: Byte): DeviceStats = nativeAllocator.getDeviceStats(device)

  /** 释放原生资源（当不再需要分配器时调用）
    */
  override def close(): Unit = {
    if (initialized()) {
      nativeAllocator.deallocate()
    }
  }

  def resetAccumulatedStats(device: Byte) = nativeAllocator.resetAccumulatedStats(device)

  def resetPeakStats(device: Byte) = nativeAllocator.resetPeakStats(device)

  /** 重置内存分配器状态
    */
  def snapshot = nativeAllocator.snapshot()

  /** 设置内存分配的增长因子
    * @param growthFactor
    *   增长因子值
    */
  def cacheInfo(device: Byte, largestBlock: SizeTPointer): Unit =
    nativeAllocator.cacheInfo(device, largestBlock)

  /** 设置内存分配的碎片整理阈值
    * @param fragmentationThreshold
    *   碎片整理阈值
    */
  def beginAllocateToPool(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair,
      filter: StreamFilter
  ): Unit =
    nativeAllocator.beginAllocateToPool(device, mempool_id, filter)

  /** 设置内存分配的最大分割大小
    * @param maxSplitSize
    *   最大分割大小（字节）
    */
  def endAllocateToPool(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair
  ): Unit = nativeAllocator.endAllocateToPool(device, mempool_id)

  /** 获取当前内存使用量（以字节为单位）
    * @param device
    *   设备索引
    * @return
    *   当前内存使用量
    */
  def releasePool(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair
  ): Unit = nativeAllocator.releasePool(device, mempool_id)

  /** 获取峰值内存使用量（以字节为单位）
    * @param device
    *   设备索引
    * @return
    *   峰值内存使用量
    */
  def getPoolUseCount(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair
  ): Int = nativeAllocator.getPoolUseCount(device, mempool_id)

  /** 清空内存缓存
    * @param device
    *   设备索引
    */
  def ensureExistsAndIncrefPool(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair
  ): Unit = nativeAllocator.ensureExistsAndIncrefPool(device, mempool_id)

  def checkPoolLiveAllocations(
      device: Byte,
      mempool_id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair,
      expected_live_allocations: PointerSet
  ) = nativeAllocator.checkPoolLiveAllocations(device, mempool_id, expected_live_allocations)

  def shareIpcHandle(ptr: Pointer) = nativeAllocator.shareIpcHandle(ptr)

  def getIpcDevPtr(handle: String) = nativeAllocator.getIpcDevPtr(handle)

  def isHistoryEnabled = nativeAllocator.isHistoryEnabled

  def recordHistory(
      enabled: Boolean,
      context_recorder: Pointer,
      alloc_trace_max_entries: Long,
      when: torch_cuda.RecordContext
  ) = nativeAllocator.recordHistory(enabled, context_recorder, alloc_trace_max_entries, when)

  def recordHistory(
      enabled: Boolean,
      context_recorder: Pointer,
      alloc_trace_max_entries: Long,
      when: Int
  ) = nativeAllocator.recordHistory(enabled, context_recorder, alloc_trace_max_entries, when)

  def recordAnnotation(md: StringPair) = nativeAllocator.recordAnnotation(md)

  def attachOutOfMemoryObserver(observer: AllocatorTraceTracker) =
    nativeAllocator.attachOutOfMemoryObserver(observer)

  def attachAllocatorTraceTracker(tracker: AllocatorTraceTracker) =
    nativeAllocator.attachAllocatorTraceTracker(tracker)

  def enablePeerAccess(dev: Byte, dev_to_access: Byte) =
    nativeAllocator.enablePeerAccess(dev, dev_to_access)

  def memcpyAsync(
      dst: Pointer,
      dstDevice: Int,
      src: Pointer,
      srcDevice: Int,
      count: Long,
      stream: CUstream_st,
      p2p_enabled: Boolean
  ): Int =
    nativeAllocator.memcpyAsync(dst, dstDevice, src, srcDevice, count, stream, p2p_enabled)

  def getCheckpointState(
      device: Byte,
      id: DeviceAssertionsDataVectorCUDAKernelLaunchInfoVectorPair
  ) = nativeAllocator.getCheckpointState(device, id)

  def setCheckpointPoolState(device: Byte, pps: AllocatorState) =
    nativeAllocator.setCheckpointPoolState(device, pps)

  def name = nativeAllocator.name()

  /** 获取分配器的调试信息
    * @return
    *   调试信息字符串
    */
  override def toString: String = s"CUDAAllocator(initialized=${initialized()})"
}

/** 伴生对象：提供工厂方法和静态访问
  */
object TorchCUDAAllocator {

  /** 获取全局 CUDA 内存分配器实例
    * @return
    *   CUDAAllocator 包装器实例
    */
  def getInstance(): CUDAAllocator = new CUDAAllocator(torch_cuda.getAllocator())

  /** 创建一个新的 CUDAAllocator 实例
    * @return
    *   CUDAAllocator 包装器实例
    */
//  def apply(): CUDAAllocator = new TorchCUDAAllocator(new CUDAAllocator())

  /** 从现有的 Java CUDAAllocator 实例创建包装器
    * @param nativeAllocator
    *   已存在的 Java 对象
    * @return
    *   包装后的 Scala 对象
    */
  def fromJava(nativeAllocator: org.bytedeco.pytorch.cuda.CUDAAllocator): CUDAAllocator =
    new CUDAAllocator(nativeAllocator)

  /** 安全地使用 CUDA 内存分配器，自动管理资源释放
    * @param f
    *   使用分配器的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行结果
    */
  def using[T](f: CUDAAllocator => T): T = {
    val allocator = getInstance()
    try {
      f(allocator)
    } finally {
      // 注意：全局分配器通常不应在此处关闭，因为它由 PyTorch 内部管理
      // 只有当使用 apply() 创建了新实例时，才需要考虑关闭
    }
  }
}
