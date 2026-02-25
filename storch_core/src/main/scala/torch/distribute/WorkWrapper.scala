package torch.distribute
import org.bytedeco.pytorch.{Future, TensorVector, TensorVectorOptional, Work}
import org.bytedeco.javacpp.{BytePointer, Pointer}
import org.bytedeco.pytorch.global.torch.*
import org.bytedeco.javacpp.chrono.Milliseconds
import org.bytedeco.pytorch.Store.kNoTimeout

/** Scala 包装器类，封装 Java 的 c10d::Work 实现 注意：原 Java 类注释提示该 API 将被 ivalue::Future 替代，使用时需谨慎
  */
class WorkWrapper private (private val underlying: Work) {

  // region 构造函数映射

  /** Pointer 构造函数（用于指针转换场景）
    */
  def this(p: Pointer) = this(new Work(p))

  /** 默认构造函数（无参数）
    */
  def this() = this(new Work())

  /** 完整参数构造函数（带默认值）
    *
    * @param rank
    *   排名（默认 -1）
    * @param opType
    *   操作类型（默认 OpType.UNKNOWN）
    * @param profilingTitle
    *   性能分析标题（默认 null）
    * @param inputTensors
    *   输入张量（默认空 optional）
    */
  def this(
      rank: Int = -1,
      opType: OpType = OpType.UNKNOWN,
      profilingTitle: String = null,
      inputTensors: TensorVectorOptional = TensorVectorOptional()
  ) = this(
    new Work(
      rank,
      opType,
      if (profilingTitle == null) null else new BytePointer(profilingTitle),
      inputTensors
    )
  )

  /** 字节类型 opType 构造函数（兼容 byte 类型参数）
    */
//  def this(
//            rank: Int = -1,
//            opType: Byte,
//            profilingTitle: String = null,
//            inputTensors: TensorVectorOptional = TensorVectorOptional.empty()
//          ) = this(
//    new Work(
//      rank,
//      opType,
//      profilingTitle,
//      inputTensors
//    )
//  )
  // endregion

  // region 实例方法映射

  /** 检查请求是否完成（非阻塞操作） */
  def isCompleted(): Boolean = underlying.isCompleted()

  /** 检查操作是否成功完成 */
  def isSuccess(): Boolean = underlying.isSuccess()

  /** 获取异常指针（若 isSuccess() 返回 false） */
  def exception(): Pointer = underlying.exception()

  /** 返回源排名（若为 recv-from-any 操作） */
  def sourceRank(): Int = underlying.sourceRank()

  /** 返回结果张量（若有） */
  def result(): TensorVector = underlying.result()

  /** 确保输出张量操作在异步工作完成后执行 */
  def synchronize(): Unit = underlying.synchronize()

  /** 阻塞等待工作完成（带超时） */
//  def wait(timeout: Milliseconds = new Milliseconds(kNoTimeout)): Boolean = underlying._wait(timeout)

  /** 中止当前工作 */
  def abort(): Unit = underlying.abort()

  /** 获取与工作完成关联的 Future 对象（仅 NCCL 后端支持） */
  def getFuture(): Future = underlying.getFuture()

  /** 获取用于跟踪工作完成状态的 Future 对象 */
  def getFutureResult(): Future = underlying.getFutureResult()

  /** 获取工作持续时间 */
  def getDuration(): Float = underlying.getDuration()

  /** 获取序列号 */
  def getSequencenumber(): Long = underlying.getSequencenumber()

  /** 获取操作类型 */
  def retrieveOpType(): OpType = underlying.retrieveOpType()
  // endregion

  /** 获取底层 Java 实例 */
//  def underlying(): Work = underlying
}

/** 伴生对象，映射静态方法 */
object WorkWrapper {

  /** 从 Future 创建 Work 实例 */
  def createFromFuture(arg0: Future): WorkWrapper =
    new WorkWrapper(Work.create_from_future(arg0))
}
