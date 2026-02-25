package torch.cuda

package torch.cuda

import org.bytedeco.cuda.cudart.CUstream_st
import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.StreamData3
import org.bytedeco.pytorch.cuda.CUDAStream
import org.bytedeco.pytorch.global.torch.DeviceType

import java.io.Closeable

/** Scala 3 包装器：封装 JavaCPP 的 CUDAStream，提供类型安全的 CUDA 流接口 用于管理 CUDA 操作的执行流，支持异步操作和资源管理
  */
class TorchCUDAStream(val nativeStream: CUDAStream) extends Closeable {

  // ------------------------------
  // 1. 基本属性访问方法
  // ------------------------------

  def asCUstream_st: CUstream_st = nativeStream.asCUstream_st()

  /** 获取原生 CUDA 流指针
    * @return
    *   CUstream_st 指针
    */
  def asStream = nativeStream.asStream()

  /** 检查流是否已创建
    * @return
    *   true 表示已创建，false 表示未创建
    */
  def device_type = nativeStream.device_type()

  /** 获取流所属的设备索引
    * @return
    *   设备索引
    */
  def device_index: Byte = nativeStream.device_index()

  def device = nativeStream.device()

  def id = nativeStream.id()

  def query = nativeStream.query()

  /** 获取流的优先级
    * @return
    *   流的优先级值
    */
  def priority(): Int = nativeStream.priority()

  // ------------------------------
  // 2. 核心功能方法
  // ------------------------------

  /** 获取流对象
    * @return
    *   CUDA 流对象
    */
  def stream: CUstream_st = nativeStream.stream()

  /** 等待流中的所有操作完成 阻塞当前线程直到流中的所有操作完成
    */
  def synchronize(): Unit = nativeStream.synchronize()

  /** 查询流是否所有操作已完成
    * @return
    *   true 表示所有操作已完成，false 表示仍有操作在进行中
    */
  def pack3: StreamData3 = nativeStream.pack3()

  def unpack3(stream_id: Long, device_index: Byte, device_type: DeviceType) =
    CUDAStream.unpack3(stream_id, device_index, device_type)

  def priority_range = CUDAStream.priority_range()

  /** 捕获流开始
    * @param mode
    *   捕获模式
    */
//  def beginCapture(mode: Int): Unit = nativeStream.beginCapture(mode)
//
//  /**
//   * 结束流捕获
//   * @return 捕获的事件
//   */
//  def endCapture(): CUDAStream = nativeStream.endCapture()
//
//  /**
//   * 等待事件完成
//   * @param event 要等待的事件
//   */
//  def waitEvent(event: CUDAStream): Unit = nativeStream.waitEvent(event)

  /** 解包 CUDA 流
    * @param stream
    *   流对象
    * @return
    *   解包后的 CUDAStream
    */
  def unwrap = nativeStream.unwrap()

  // ------------------------------
  // 3. 资源管理（实现 Closeable 接口）
  // ------------------------------

  /** 获取流的原生指针（用于底层操作）
    * @return
    *   指向 CUDA 流的原生指针
    */
  def address(): Long = nativeStream.address()

  /** 检查流是否为空（未分配原生资源）
    * @return
    *   true 表示未分配资源
    */
  def isNull(): Boolean = nativeStream.isNull()

  /** 释放原生资源（等价于销毁 CUDA 流）
    */
  def deallocate(): Unit = nativeStream.deallocate()

  /** 关闭资源（自动释放原生内存，建议配合 try-with-resources 使用）
    */
  override def close(): Unit = {
    if (!nativeStream.isNull()) nativeStream.deallocate()
  }

  // ------------------------------
  // 4. 辅助方法
  // ------------------------------

  /** 检查当前流是否与另一个流指向相同的原生资源
    * @param other
    *   另一个流
    * @return
    *   true 表示指向相同原生资源
    */
  def eq(other: CUDAStream): Boolean = nativeStream eq other

  /** 获取原生资源的哈希码（用于集合操作）
    * @return
    *   原生指针的哈希值
    */
  override def hashCode(): Int = nativeStream.hashCode()

  /** 字符串表示（包含原生指针地址）
    * @return
    *   格式："TorchCUDAStream(pointer=0xXXXXXXXX, device_index=X)"
    */
  override def toString: String =
    s"TorchCUDAStream(pointer=0x${address().toHexString}, device_index)"
}

/** 伴生对象：提供工厂方法和静态访问
  */
object TorchCUDAStream {

  // CUDA 流的默认优先级
  val DEFAULT_PRIORITY: Int = 0

  /** 创建一个默认的 CUDA 流 // * @return TorchCUDAStream 包装器实例 //
    */
//  def apply(): TorchCUDAStream = new TorchCUDAStream(new CUDAStream())
//
//  /**
//   * 指定设备创建 CUDA 流
//   * @param device_index 设备索引
//   * @return TorchCUDAStream 包装器实例
//   */
//  def apply(device_index: Byte): TorchCUDAStream = new TorchCUDAStream(new CUDAStream(device_index))

  /** 指定设备和优先级创建 CUDA 流
    * @param device_index
    *   设备索引
    * @param priority
    *   流的优先级，值越小优先级越高
    * @return
    *   TorchCUDAStream 包装器实例
    */
//  def apply(device_index: Byte, priority: Int): TorchCUDAStream = new TorchCUDAStream(new CUDAStream(device_index, priority))

  /** 从已有的 Java CUDAStream 实例包装
    * @param nativeStream
    *   已存在的 Java 对象
    * @return
    *   包装后的 Scala 对象
    */
//  def fromJava(nativeStream: CUDAStream): TorchCUDAStream = new TorchCUDAStream(nativeStream)
//
//  /**
//   * 安全地使用 CUDAStream，自动管理资源释放
//   * @param deviceIndex 要使用的 CUDA 设备索引
//   * @param f 在指定流上下文中执行的函数
//   * @tparam T 返回类型
//   * @return 函数执行结果
//   */
//  def using[T](deviceIndex: Byte)(f: TorchCUDAStream => T): T = {
//    val stream = TorchCUDAStream(deviceIndex)
//    try {
//      f(stream)
//    } finally {
//      stream.close()
//    }
//  }

  /** 安全地使用默认 CUDAStream，自动管理资源释放
    * @param f
    *   在默认流上下文中执行的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行结果
    */
//  def usingDefault[T](f: TorchCUDAStream => T): T = {
//    val stream = TorchCUDAStream()
//    try {
//      f(stream)
//    } finally {
//      stream.close()
//    }
//  }

  /** 安全地使用指定优先级的 CUDAStream，自动管理资源释放
    * @param deviceIndex
    *   设备索引
    * @param priority
    *   流优先级
    * @param f
    *   在指定流上下文中执行的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行结果
    */
//  def usingWithPriority[T](deviceIndex: Byte, priority: Int)(f: TorchCUDAStream => T): T = {
//    val stream = TorchCUDAStream(deviceIndex, priority)
//    try {
//      f(stream)
//    } finally {
//      stream.close()
//    }
//  }

  /** 获取当前线程的默认 CUDA 流
    * @return
    *   当前默认流的包装器
    */
//  def getCurrentStream: TorchCUDAStream = fromJava(cuda.getCurrentCUDAStream)
//
//  /**
//   * 获取默认 CUDA 流
//   * @return 默认流的包装器
//   */
//  def getDefaultStream: TorchCUDAStream = fromJava(cuda.getDefaultCUDAStream)
//
//  /**
//   * 设置当前 CUDA 流
//   * @param stream 要设置的流
//   */
//  def setCurrentStream(stream: TorchCUDAStream): Unit = cuda.setCurrentCUDAStream(stream.nativeStream)

  /** 打包 CUDA 流（静态方法）
    * @param stream
    *   流对象
    * @return
    *   打包后的 CUDAStream
    */
//  def pack3(stream: CUstream_st): CUDAStream = CUDAStream.pack3(stream)

  /** 解包 CUDA 流（静态方法）
    * @param stream
    *   CUDAStream 对象
    * @return
    *   解包后的 CUstream_st 指针
    */
//  def unpack3(stream: CUDAStream): CUstream_st = CUDAStream.unpack3(stream)
}
