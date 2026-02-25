/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch

import org.bytedeco.pytorch.ProcessGroup
import org.bytedeco.pytorch.cuda.CUDAStream
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.global.torch_cuda as torchCuda
import torch.distribute.DistBackend
import org.bytedeco.javacpp.*
import org.bytedeco.cuda.cudart.*
import org.bytedeco.cuda.global.cudart.cudaDeviceReset

/** This package adds support for CUDA tensor types, that implement the same function as CPU
  * tensors, but they utilize GPUs for computation.
  */
package object cuda {

  /** Returns a Boolean indicating if CUDA is currently available. */
  def isAvailable: Boolean = torchNative.cuda_is_available()

  def is_available: Boolean = torchNative.cuda_is_available()

  def is_available(un_used: Int*): Boolean = torchNative.cuda_is_available()

  def device_count = torchNative.cuda_device_count()

  def device_count(un_used: Int*) = torchNative.cuda_device_count()

  def cudnn_is_available: Boolean = torchNative.cudnn_is_available()
//
//  def getCurrentCUDASolverDnHandle = torchCuda.getCurrentCUDASolverDnHandle()
//
//  def getCudnnHandle = torchCuda.getCudnnHandle()

  def empty_cache(un_used: Int*): Unit = cudaDeviceReset()

// udaPointerGetAttributes(cudaPointerAttributes attributes, @Const Pointer ptr);
  // public static native @Cast("CUresult") int cuMemGetInfo(@Cast("size_t*") SizeTPointer _free, @Cast("size_t*") SizeTPointer total);
//  def memory_allocated(device: Device) = torchCuda.memory_allocated(device)

// Cast("cudaError_t") int cudaDeviceSetMemPool(int device, CUmemPoolHandle_st memPool);
//int cudaDeviceGetMemPool(@ByPtrPtr CUmemPoolHandle_st memPool, int device);
  //      val memInfo = new cudaDeviceProp()
  //      cudaGetDeviceProperties(memInfo, id)
  //      memInfo.totalGlobalMem()
  //  def memory_reserved(device: Device) = torchCuda.memory_reserved(device)

  def current_device = torchCuda.current_device()

  def current_device(un_used: Int*) = torchCuda.current_device()

  def device_count_ensure_non_zero = torchCuda.device_count_ensure_non_zero()

  def make_generator_cuda = torchCuda.make_generator_cuda()

  def set_target_device = torchCuda.SetTargetDevice()

  def MaybeExchangeDevice(device: Byte) = torchCuda.MaybeExchangeDevice(device)

  def ExchangeDevice(device: Byte) = torchCuda.ExchangeDevice(device)

  def get_device(device: String): Int = {
    val pointer = new BytePointer(device)
    torchCuda.GetDevice(pointer)
  }

  def getCurrentDeviceProperties = torchCuda.getCurrentDeviceProperties()

  def GetDeviceCount(devs: Array[Int]) = torchCuda.GetDeviceCount(devs)

  def SetDevice(device: Byte) = torchCuda.SetDevice(device)

  def MaybeSetDevice(device: Byte) = torchCuda.MaybeSetDevice(device)

  def hasPrimaryContext(device_index: Byte) = torchCuda.hasPrimaryContext(device_index)
//  def get_device_name(device: Int) = torchCuda.get_device_name(device)

  def clearCublasWorkspaces = torchCuda.clearCublasWorkspaces()

  def getDeviceIndexWithPrimaryContext() = torchCuda.getDeviceIndexWithPrimaryContext()

  def currentStreamCaptureStatusMayInitCtx = torchCuda.currentStreamCaptureStatusMayInitCtx()

  def manual_seed(seed: Long) = torchNative.cuda_manual_seed(seed)

  def manual_seed_all(seed: Long) = torchNative.cuda_manual_seed_all(seed)

  def cuda_manual_seed(seed: Long) = torchNative.cuda_manual_seed(seed)

  def cuda_manual_seed_all(seed: Long) = torchNative.cuda_manual_seed_all(seed)

  def cuda_synchronize = torchNative.cuda_synchronize()

  def cuda_synchronize(sync: Long) = torchNative.cuda_synchronize(sync)

  def CUDA_HELP = torchNative.CUDA_HELP()

  def getCUDAHooks = torchNative.getCUDAHooks()

  def set_device(device: Byte) = torchCuda.set_device(device)

  def device_synchronize = torchCuda.device_synchronize()

  def warn_or_error_on_sync = torchCuda.warn_or_error_on_sync()

  def getNumGPUs = torchCuda.getNumGPUs()

  def get_num_gpus = torchCuda.getNumGPUs()

  def warp_size = torchCuda.warp_size()

  def getDefaultCUDAStream = torchCuda.getDefaultCUDAStream()

  def getCurrentCUDAStream = torchCuda.getCurrentCUDAStream()

  def setCurrentCUDAStream(stream: CUDAStream) = torchCuda.setCurrentCUDAStream(stream)

  def dataSize(dtype: Int) = torchCuda.dataSize(dtype)

  def getAllocator = torchCuda.getAllocator()

  def getMemoryFraction(frac: Byte) = torchCuda.getMemoryFraction(frac)

  def isEnabled: Boolean = torchCuda.isEnabled()

  def is_enabled: Boolean = torchCuda.isEnabled()

  def enable(flag: Boolean) = torchCuda.enable(flag)

  def contiguousIfZeroInStrides[D <: DType](tensor: Tensor[D]) = fromNative(
    torchCuda.contiguousIfZeroInStrides(tensor.native)
  )
}
