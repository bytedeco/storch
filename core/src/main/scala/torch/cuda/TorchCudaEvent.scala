package torch.cuda

import org.bytedeco.cuda.cudart.{CUevent_st, cudaIpcEventHandle_t}
import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.cuda.{CUDAEvent, CUDAStream}

import java.io.Closeable

/** Scala 3 包装器：完整委托 Java CUDAEvent 的所有方法 封装 CUDA 事件的创建、记录、同步、查询等核心操作
  */
class TorchCudaEvent(val nativeEvent: CUDAEvent) extends Closeable {

  // ------------------------------
  // 1. 构造函数相关（通过伴生对象工厂方法创建）
  // ------------------------------

  def asCUevent_st = nativeEvent.asCUevent_st()

  def isCreated = nativeEvent.isCreated()

  def device_index = nativeEvent.device_index()
  // ------------------------------
  // 2. 核心功能方法（严格对应 Java 类方法）
  // ------------------------------

  def event: CUevent_st = nativeEvent.event()

  /** 查询事件是否已完成
    *
    * @return
    *   true 表示事件已完成，false 表示未完成
    * @see
    *   CUDAEvent#query()
    */
  def query(): Boolean = nativeEvent.query()

  /** 在指定 CUDA 流上记录事件
    * @param stream
    *   目标流（若为 null 则使用默认流）
    * @see
    *   CUDAEvent#record(CUDAStream)
    */
  def record(stream: CUDAStream): Unit = nativeEvent.record(stream)

  def record: Unit = nativeEvent.record()

  def recordOnce(stream: CUDAStream): Unit = nativeEvent.recordOnce(stream)

  /** 阻塞等待事件完成
    * @see
    *   CUDAEvent#synchronize()
    */
  def synchronize(): Unit = nativeEvent.synchronize()

  /** 计算两个事件之间的耗时（毫秒）
    * @param other
    *   另一个事件（需已记录）
    * @return
    *   时间差（毫秒，精度 ~0.5 微秒）
    * @see
    *   CUDAEvent#elapsedTime(CUDAEvent)
    */
  def elapsed_time(other: CUDAEvent): Float = nativeEvent.elapsed_time(other)

  /** 获取事件创建时的标志位
    * @return
    *   构造函数传入的 flags 参数
    * @see
    *   CUDAEvent#flags()
    */
  def ipc_handle(handle: cudaIpcEventHandle_t) = nativeEvent.ipc_handle(handle)

  def device = nativeEvent.device()

//  def lessThan(left: CUDAEvent, right: CUDAEvent) = nativeEvent.lessThan(left, right)

  def lessThan(right: CUDAEvent) = nativeEvent.lessThan(right)

  /** 获取事件的原生指针（用于底层操作）
    * @return
    *   指向 CUDA 事件的原生指针
    * @see
    *   CUDAEvent#address()
    */
  def address(): Long = nativeEvent.address()

  def block(stream: CUDAStream) = nativeEvent.block(stream)

  /** 检查事件是否为空（未分配原生资源）
    * @return
    *   true 表示未分配资源
    * @see
    *   CUDAEvent#isNull()
    */
  def isNull(): Boolean = nativeEvent.isNull()

  /** 释放原生资源（等价于销毁 CUDA 事件）
    * @see
    *   CUDAEvent#deallocate()
    */
  def deallocate(): Unit = nativeEvent.deallocate()

  // ------------------------------
  // 3. 资源管理（实现 Closeable 接口）
  // ------------------------------

  /** 关闭资源（自动释放原生内存，建议配合 try-with-resources 使用）
    * @see
    *   Closeable#close()
    */
  override def close(): Unit = {
    if (!nativeEvent.isNull()) nativeEvent.deallocate()
  }

  // ------------------------------
  // 4. 辅助方法（基于 Java 类的扩展功能）
  // ------------------------------

  /** 检查当前事件是否与另一个事件指向相同的原生资源
    * @param other
    *   另一个事件包装器
    * @return
    *   true 表示指向相同原生资源
    */
  def eq(other: CUDAEvent): Boolean = nativeEvent eq other

  /** 获取原生资源的哈希码（用于集合操作）
    * @return
    *   原生指针的哈希值
    */
  override def hashCode(): Int = nativeEvent.hashCode()

  /** 字符串表示（包含原生指针地址）
    * @return
    *   格式："CUDAEventWrapper(pointer=0xXXXXXXXX)"
    */
  override def toString: String = s"CUDAEventWrapper(pointer=0x${address().toHexString})"
}

/** 伴生对象：提供工厂方法，对应 Java 类的构造函数
  */
object TorchCudaEvent {

  /** 创建默认标志的 CUDA 事件
    * @see
    *   CUDAEvent#CUDAEvent()
    */
  def apply(): TorchCudaEvent = new TorchCudaEvent(new CUDAEvent())

  /** 指定标志创建 CUDA 事件
    * @param flags
    *   事件创建标志（如 CUDAEvent.DEFAULT、CUDAEvent.BLOCKING_SYNC 等）
    * @see
    *   CUDAEvent#CUDAEvent(int)
    */
  def apply(flags: Int): TorchCudaEvent = new TorchCudaEvent(new CUDAEvent(flags))

  /** 从已有的 Java CUDAEvent 实例包装
    * @param nativeEvent
    *   已存在的 Java 对象
    * @return
    *   包装后的 Scala 对象
    */
  def fromJava(nativeEvent: CUDAEvent): TorchCudaEvent = new TorchCudaEvent(nativeEvent)
}
