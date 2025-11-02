package torch.profiler

import org.bytedeco.pytorch.{ExperimentalConfig}

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.StringVector

class ExperimentalConfigWrapper private (private val underlying: ExperimentalConfig) {

  def this(
      profilerMetrics: StringVector = new StringVector(),
      profilerMeasurePerKernel: Boolean = false,
      verbose: Boolean = false,
      performanceEvents: StringVector = new StringVector(),
      enableCudaSyncEvents: Boolean = false,
      adjustProfilerStep: Boolean = false,
      disableExternalCorrelation: Boolean = false,
      profileAllThreads: Boolean = false,
      captureOverloadNames: Boolean = false,
      adjustTimestamps: Boolean = false
  ) = this(
    new ExperimentalConfig(
      profilerMetrics,
      profilerMeasurePerKernel,
      verbose,
      performanceEvents,
      enableCudaSyncEvents,
      adjustProfilerStep,
      disableExternalCorrelation,
      profileAllThreads,
      captureOverloadNames,
      adjustTimestamps
    )
  )

  def this() = this(new ExperimentalConfig())

  def this(p: Pointer) = this(new ExperimentalConfig(p))

  def this(size: Long) = this(new ExperimentalConfig(size))

  def asBoolean(): Boolean = underlying.asBoolean()

  def position(position: Long): ExperimentalConfigWrapper = {
    underlying.position(position)
    this
  }

  def getPointer(i: Long): ExperimentalConfigWrapper =
    new ExperimentalConfigWrapper(underlying.getPointer(i))

  def profilerMetrics(): StringVector = underlying.profiler_metrics()

  def profilerMetrics_=(setter: StringVector): Unit = underlying.profiler_metrics(setter)

  def performanceEvents(): StringVector = underlying.performance_events()

  def performanceEvents_=(setter: StringVector): Unit = underlying.performance_events(setter)

  def profilerMeasurePerKernel(): Boolean = underlying.profiler_measure_per_kernel()

  def profilerMeasurePerKernel_=(setter: Boolean): Unit =
    underlying.profiler_measure_per_kernel(setter)

  def verbose(): Boolean = underlying.verbose()

  def verbose_=(setter: Boolean): Unit = underlying.verbose(setter)

  def enableCudaSyncEvents(): Boolean = underlying.enable_cuda_sync_events()

  def enableCudaSyncEvents_=(setter: Boolean): Unit = underlying.enable_cuda_sync_events(setter)

  def adjustProfilerStep(): Boolean = underlying.adjust_profiler_step()

  def adjustProfilerStep_=(setter: Boolean): Unit = underlying.adjust_profiler_step(setter)

  def disableExternalCorrelation(): Boolean = underlying.disable_external_correlation()

  def disableExternalCorrelation_=(setter: Boolean): Unit =
    underlying.disable_external_correlation(setter)

  def profileAllThreads(): Boolean = underlying.profile_all_threads()

  def profileAllThreads_=(setter: Boolean): Unit = underlying.profile_all_threads(setter)

  def captureOverloadNames(): Boolean = underlying.capture_overload_names()

  def captureOverloadNames_=(setter: Boolean): Unit = underlying.capture_overload_names(setter)

  def adjustTimestamps(): Boolean = underlying.adjust_timestamps()

  def adjustTimestamps_=(setter: Boolean): Unit = underlying.adjust_timestamps(setter)

//  def underlying(): ExperimentalConfig = underlying
}
