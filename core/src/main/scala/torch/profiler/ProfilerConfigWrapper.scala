package torch.profiler

import org.bytedeco.pytorch.{ProfilerConfig, ExperimentalConfig, IValue}

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.pytorch.global.torch._

class ProfilerConfigWrapper(private val underlying: ProfilerConfig) {

  def this(
      state: ProfilerState,
      reportInputShapes: Boolean = false,
      profileMemory: Boolean = false,
      withStack: Boolean = false,
      withFlops: Boolean = false,
      withModules: Boolean = false,
      experimentalConfig: ExperimentalConfig = new ExperimentalConfig(),
      traceId: String = ""
  ) = this(
    new ProfilerConfig(
      state,
      reportInputShapes,
      profileMemory,
      withStack,
      withFlops,
      withModules,
      experimentalConfig,
      new BytePointer(traceId)
    )
  )

  def this(state: ProfilerState) = this(new ProfilerConfig(state))

  def this(state: Int) = this(new ProfilerConfig(state))

  def disabled(): Boolean = underlying.disabled()

  def global(): Boolean = underlying.global()

  def pushGlobalCallbacks(): Boolean = underlying.pushGlobalCallbacks()

  def state(): ProfilerState = underlying.state()

  def state_=(setter: ProfilerState): Unit = underlying.state(setter)

  def experimentalConfig(): ExperimentalConfig = underlying.experimental_config()

  def experimentalConfig_=(setter: ExperimentalConfig): Unit =
    underlying.experimental_config(setter)

  def reportInputShapes(): Boolean = underlying.report_input_shapes()

  def reportInputShapes_=(setter: Boolean): Unit = underlying.report_input_shapes(setter)

  def profileMemory(): Boolean = underlying.profile_memory()

  def profileMemory_=(setter: Boolean): Unit = underlying.profile_memory(setter)

  def withStack(): Boolean = underlying.with_stack()

  def withStack_=(setter: Boolean): Unit = underlying.with_stack(setter)

  def withFlops(): Boolean = underlying.with_flops()

  def withFlops_=(setter: Boolean): Unit = underlying.with_flops(setter)

  def withModules(): Boolean = underlying.with_modules()

  def withModules_=(setter: Boolean): Unit = underlying.with_modules(setter)

  def traceId(): String = underlying.trace_id().getString

  def traceId_=(setter: String): Unit = underlying.trace_id(new BytePointer(setter))

  def toIValue(): IValue = underlying.toIValue()

//  def underlying(): ProfilerConfig = underlying
}

object ProfilerConfigWrapper {

  def fromIValue(profilerConfigIValue: IValue): ProfilerConfigWrapper =
    new ProfilerConfigWrapper(ProfilerConfig.fromIValue(profilerConfigIValue))
}
