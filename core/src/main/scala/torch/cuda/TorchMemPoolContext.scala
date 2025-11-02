package torch.cuda

import org.bytedeco.pytorch.cuda.{MemPoolContext => NativeMemPoolContext}
import org.bytedeco.javacpp.Pointer
import java.io.Closeable

/** Scala 3 包装器：封装 JavaCPP 的 MemPoolContext，提供类型安全的 CUDA 内存池上下文接口 管理 CUDA 内存池的创建、配置和生命周期
  */
class TorchMemPoolContext(val nativeContext: NativeMemPoolContext) extends Closeable {

  // ------------------------------
  // 1. 构造函数相关（通过伴生对象工厂方法创建）
  // ------------------------------

  /** 从指针创建 MemPoolContext 实例
    * @param p
    *   指向原生 MemPoolContext 的指针
    */
  def this(p: Pointer) = this(new NativeMemPoolContext(p))

  // ------------------------------
  // 2. 核心功能方法
  // ------------------------------

  /** 获取指向原生 MemPoolContext 的指针地址
    * @return
    *   指针地址
    */
  def address(): Long = nativeContext.address()

  /** 检查上下文是否为空
    * @return
    *   如果为空返回 true，否则返回 false
    */
  def isNull(): Boolean = nativeContext.isNull()

  /** 关闭资源，释放内存池上下文
    */
  override def close(): Unit = nativeContext.close()

  /** 比较两个 MemPoolContext 是否引用同一个原生对象
    * @param other
    *   另一个 NativeMemPoolContext 对象
    * @return
    *   如果引用相同对象则返回 true
    */
  def eq(other: NativeMemPoolContext): Boolean = nativeContext eq other

  // ------------------------------
  // 3. 内存池相关方法
  // ------------------------------

  /** 获取内存池设备索引
    * @return
    *   设备索引
    */
  def getActiveMemPool = NativeMemPoolContext.getActiveMemPool()

  /** 激活内存池上下文
    */
//  def activate(): Unit = nativeContext.activate()
//
//  /**
//   * 停用内存池上下文
//   */
//  def deactivate(): Unit = nativeContext.deactivate()
//
//  /**
//   * 重置内存池上下文到初始状态
//   */
//  def reset(): Unit = nativeContext.reset()
//
//  /**
//   * 检查内存池是否已激活
//   * @return 如果已激活返回 true
//   */
//  def isActivated(): Boolean = nativeContext.isActivated()

  // ------------------------------
  // 4. 字符串表示
  // ------------------------------

  /** 返回内存池上下文的字符串表示
    * @return
    *   格式："TorchMemPoolContext(pointer=0xXXXXXXXX, device_index=X)"
    */
  override def toString: String =
    s"TorchMemPoolContext(pointer=0x${address().toHexString}, device_index=)"
}

/** TorchMemPoolContext 伴生对象，提供工厂方法和实用工具
  */
object TorchMemPoolContext {

  /** 创建一个新的内存池上下文
    * @param deviceIndex
    *   设备索引
    * @return
    *   MemPoolContext 包装器实例
    */
//  def apply(deviceIndex: Byte): TorchMemPoolContext =
//    new TorchMemPoolContext(new NativeMemPoolContext(deviceIndex))

  /** 从现有的 Java NativeMemPoolContext 实例创建包装器
    * @param nativeContext
    *   原生 MemPoolContext 实例
    * @return
    *   MemPoolContext 包装器实例
    */
  def fromJava(nativeContext: NativeMemPoolContext): TorchMemPoolContext =
    new TorchMemPoolContext(nativeContext)

  /** 安全地使用 MemPoolContext，自动管理资源释放
    * @param deviceIndex
    *   设备索引
    * @param f
    *   要在 MemPoolContext 上下文中执行的函数
    * @tparam T
    *   返回类型
    * @return
    *   函数执行的结果
    */
//  def using[T](deviceIndex: Byte)(f: TorchMemPoolContext => T): T = {
//    val context = TorchMemPoolContext(deviceIndex)
//    try {
//      context.activate()
//      f(context)
//    } finally {
//      context.deactivate()
//      context.close()
//    }
//  }
//
//  /**
//   * 创建默认的内存池上下文
//   * @return MemPoolContext 包装器实例
//   */
//  def defaultContext(): TorchMemPoolContext =
//    new TorchMemPoolContext(new NativeMemPoolContext())
}
