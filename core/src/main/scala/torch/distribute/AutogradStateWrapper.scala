package torch.distribute

import org.bytedeco.pytorch.{AutogradState}

import org.bytedeco.javacpp.Pointer

/** Scala 包装器类，封装 Java 的 AutogradState 实现 映射所有静态方法、构造函数及布尔标志的 get/set 方法
  */
class AutogradStateWrapper private (private val underlying: AutogradState) {

  // region 构造函数映射
  /** Pointer 构造函数（用于指针转换场景）
    */
  def this(p: Pointer) = this(new AutogradState(p))

  /** 完整参数构造函数（映射 Java 的四参数构造函数）
    */
  def this(
      gradMode: Boolean,
      inferenceMode: Boolean,
      fwGradMode: Boolean,
      multithreadingEnabled: Boolean
  ) = this(
    new AutogradState(gradMode, inferenceMode, fwGradMode, multithreadingEnabled)
  )
  // endregion

  // region 标志设置方法（setter）
  def setGradMode(enabled: Boolean): Unit = underlying.set_grad_mode(enabled)

  def setFwGradMode(enabled: Boolean): Unit = underlying.set_fw_grad_mode(enabled)

  def setInferenceMode(enabled: Boolean): Unit = underlying.set_inference_mode(enabled)

  def setMultithreadingEnabled(enabled: Boolean): Unit =
    underlying.set_multithreading_enabled(enabled)

  def setViewReplayEnabled(enabled: Boolean): Unit = underlying.set_view_replay_enabled(enabled)
  // endregion

  // region 标志获取方法（getter）
  def getGradMode(): Boolean = underlying.get_grad_mode()

  def getFwGradMode(): Boolean = underlying.get_fw_grad_mode()

  def getInferenceMode(): Boolean = underlying.get_inference_mode()

  def getMultithreadingEnabled(): Boolean = underlying.get_multithreading_enabled()

  def getViewReplayEnabled(): Boolean = underlying.get_view_replay_enabled()
  // endregion

  /** 获取底层 Java 实例（用于与其他 Java API 交互）
    */
//  def underlying(): AutogradState = underlying
}

/** 伴生对象，映射 Java 静态方法
  */
object AutogradStateWrapper {

  /** 获取线程本地状态（映射 get_tls_state()）
    */
  def getTlsState(): AutogradStateWrapper =
    new AutogradStateWrapper(AutogradState.get_tls_state())

  /** 设置线程本地状态（映射 set_tls_state()）
    */
  def setTlsState(state: AutogradStateWrapper): Unit =
    AutogradState.set_tls_state(state.underlying)
}
