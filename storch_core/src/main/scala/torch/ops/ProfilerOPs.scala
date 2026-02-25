package torch
package ops

import org.bytedeco.pytorch.{
  StringViewOptional,
  TensorOptions,
  LongArrayRefOptional,
  TensorOptional,
  LongOptional,
  BoolOptional,
  SymInt,
  StringIValueMap,
  SymIntOptional,
  LongVector
}
import Layout.Strided
import Device.CPU
import internal.NativeConverters
import NativeConverters.*
import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.annotation.{ByRef, ByVal, Const, Namespace}

trait ProfilerOPs {

  def in_parallel_region = torchNative.in_parallel_region()

  def lazy_init_num_threads = torchNative.lazy_init_num_threads()

  def set_thread_num(num_threads: Int) = torchNative.set_thread_num(num_threads)

  def get_parallel_info = torchNative.get_parallel_info()

  def set_num_interop_threads(num_threads: Int) = torchNative.set_num_interop_threads(num_threads)

  def get_num_interop_threads = torchNative.get_num_interop_threads()

  def intraop_default_num_threads = torchNative.intraop_default_num_threads()

  def actToString(act: Int) = torchNative.actToString(act)

  def profilerEnabled = torchNative.profilerEnabled()

  def profilerType = torchNative.profilerType()

  def getProfilerConfig = torchNative.getProfilerConfig()

  def softAssertRaises = torchNative.softAssertRaises()

  def ProfilerPerfEvents = torchNative.ProfilerPerfEvents()

  def computeFlops(name: String, sn_map: StringIValueMap) = torchNative.computeFlops(name, sn_map)

  def isProfilerEnabledInMainThread = torchNative.isProfilerEnabledInMainThread()

  def enableProfilerInChildThread = torchNative.enableProfilerInChildThread()

  def disableProfilerInChildThread = torchNative.disableProfilerInChildThread()

}
