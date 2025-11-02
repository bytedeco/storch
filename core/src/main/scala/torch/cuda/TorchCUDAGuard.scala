package torch.cuda

import org.bytedeco.pytorch.cuda.{CUDAGuard, CUDAGuard as NativeCUDAGuard}
import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.Device

import java.io.Closeable

/** Scala 3 包装器：封装 JavaCPP 的 CUDAGuard，提供类型安全的 CUDA 设备守卫接口 用于临时切换 CUDA 设备并在作用域结束时自动恢复原始设备
  *
  * 类似于 PyTorch 中的 `with torch.cuda.device(device)` 上下文管理器
  */
class TorchCUDAGuard(val nativeGuard: NativeCUDAGuard) extends Closeable {

  def set_device(device: Device): Unit = nativeGuard.set_device(device)

  def reset_device(device: Device): Unit = nativeGuard.reset_device(device)

  /** 获取当前守卫的 CUDA 设备索引
    * @return
    *   设备索引
    */
  def set_index(device_index: Byte): Unit = nativeGuard.set_index(device_index)

  /** 检查守卫是否已创建（资源是否已分配）
    * @return
    *   true 表示已创建，false 表示未创建
    */
  def original_device() = nativeGuard.original_device()

  /** 重置守卫到默认状态
    */
  def current_device() = nativeGuard.current_device()

  /** 释放原生资源（当不再需要守卫时调用） 在 Scala 中通常通过 try-with-resources 或自动资源管理使用
    */
  override def close(): Unit = {

    nativeGuard.deallocate()

  }

  /** 获取原生指针（用于底层操作）
    * @return
    *   指向 CUDAGuard 的原生指针
    */
  def address(): Long = nativeGuard.address()

  /** 检查守卫是否为空（未分配原生资源）
    * @return
    *   true 表示未分配资源
    */
  def isNull(): Boolean = nativeGuard.isNull()

  /** 字符串表示（包含设备索引信息）
    * @return
    *   格式："CUDAGuard(device_index=X)"
    */
  override def toString: String = s"CUDAGuard(device_index=${current_device().index()})"

  /** 比较当前守卫与另一个守卫是否指向相同的原生资源
    * @param other
    *   另一个守卫
    * @return
    *   true 表示指向相同原生资源
    */
  def eq(other: NativeCUDAGuard): Boolean = nativeGuard eq other

  /** 获取原生资源的哈希码（用于集合操作）
    * @return
    *   原生指针的哈希值
    */
  override def hashCode(): Int = nativeGuard.hashCode()
}

/** 伴生对象：提供工厂方法和静态访问
  */
object TorchCUDAGuard {

  /** 创建一个守卫指定 CUDA 设备的 CUDAGuard 实例
    * @param deviceIndex
    *   要守卫的 CUDA 设备索引
    * @return
    *   CUDAGuard 包装器实例
    */
  def apply(deviceIndex: Byte): CUDAGuard = new CUDAGuard(new NativeCUDAGuard(deviceIndex))

  /** 创建一个守卫当前设备的 CUDAGuard 实例
    * @return
    *   CUDAGuard 包装器实例
    */
//  def apply(): CUDAGuard = new CUDAGuard(new NativeCUDAGuard())

  /** 从现有的 Java NativeCUDAGuard 实例创建包装器
    * @param nativeGuard
    *   已存在的 Java 对象
    * @return
    *   包装后的 Scala 对象
    */
  def fromJava(nativeGuard: NativeCUDAGuard): CUDAGuard = new CUDAGuard(nativeGuard)

  /** 安全地使用 CUDAGuard，自动管理资源释放
    * @param deviceIndex
    *   要使用的 CUDA 设备索引
    * @param f
    *   在指定设备上下文中执行的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行结果
    */
  def using[T](deviceIndex: Byte)(f: => T): T = {
    val guard = CUDAGuard(deviceIndex)
    try {
      f
    } finally {
      guard.close()
    }
  }

  /** 安全地使用 CUDAGuard 守卫当前设备，自动管理资源释放
    * @param f
    *   在当前设备上下文中执行的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行结果
    */
//  def usingCurrent[T](f: => T): T = {
//    val guard = TorchCUDAGuard()
//    try {
//      f
//    } finally {
//      guard.close()
//    }
//  }
}
