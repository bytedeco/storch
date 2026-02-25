package torch.profiler

import org.bytedeco.pytorch.ExperimentalConfig
import org.bytedeco.javacpp.{BytePointer, Pointer}
import org.bytedeco.pytorch.StringVector

class ExperimentalConfigWrapper private (private val underlying: ExperimentalConfig) {

  def this(
            profiler_metrics: StringVector = new StringVector(),
      profiler_measure_per_kernel: Boolean = false,
      verbose: Boolean = false,
      performance_events: StringVector = new StringVector(),
      enable_cuda_sync_events: Boolean = false,
      adjust_profiler_step: Boolean = false,
      disable_external_correlation: Boolean = false,
      profile_all_threads: Boolean = false,
      capture_overload_names: Boolean = false,
      record_python_gc_info: Boolean = false,
      expose_kineto_event_metadata: Boolean = false,
      custom_profiler_config: String = "",
      adjust_timestamps: Boolean = false
  ) = this(
    new ExperimentalConfig(
      profiler_metrics,
      profiler_measure_per_kernel,
      verbose,
      performance_events,
      enable_cuda_sync_events,
      adjust_profiler_step,
      disable_external_correlation,
      profile_all_threads,
      capture_overload_names,
      record_python_gc_info,
      expose_kineto_event_metadata,
      new BytePointer(custom_profiler_config),
      adjust_timestamps
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
