package torch
package utils
package data
package dataloader

import org.bytedeco.javacpp.chrono.{Milliseconds, Seconds}
import org.bytedeco.pytorch.{
  ChunkDataset,
  ChunkDatasetOptions,
  ChunkMapDataset,
  ChunkRandomDataLoader,
  ChunkSharedBatchDataset,
  Example,
  ExampleIterator,
  ExampleStack,
  ExampleVector
}
import torch.{DType, Default}
import torch.utils.data.sampler.Sampler

import scala.collection.mutable.ArrayBuffer

trait TorchDataLoader

case class TorchDataLoaderOptions(
    batch_size: Int = 1,
    shuffle: Boolean = true,
    sampler: Sampler = null,
    batch_sampler: Sampler = null,
    num_workers: Int = 0,
    max_jobs: Long = 0,
    collate_fn: Any = null,
    pin_memory: Boolean = false,
    drop_last: Boolean = false,
    in_order: Boolean = false,
    timeout: Float = 0,
    worker_init_fn: Any = null,
    prefetch_factor: Option[Int] = Some(2),
    persistent_workers: Boolean = false
) {
  def toNative: org.bytedeco.pytorch.DataLoaderOptions = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(batch_size)
    loaderOpts.batch_size.put(batch_size)
    loaderOpts.drop_last().put(drop_last)
    loaderOpts.enforce_ordering().put(in_order)
    loaderOpts.workers().put(num_workers)
    loaderOpts.max_jobs().put(max_jobs)
    //   loaderOpts.timeout(new Milliseconds(new Seconds(timeout.toLong)))
//    loaderOpts.timeout().put(new Milliseconds(timeout.toLong)) //todo javacpp bug ,wait to fix
    loaderOpts
  }

  def toNativeFull: org.bytedeco.pytorch.FullDataLoaderOptions = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(batch_size)
    loaderOpts.batch_size.put(batch_size)
    loaderOpts.drop_last().put(drop_last)
    loaderOpts.enforce_ordering().put(in_order)
    loaderOpts.workers().put(num_workers)
    loaderOpts.max_jobs().put(max_jobs)
    //   loaderOpts.timeout(new Milliseconds(new Seconds(timeout.toLong)))
//    loaderOpts.timeout().put(new Milliseconds(timeout.toLong)) //todo javacpp bug ,wait to fix
    new org.bytedeco.pytorch.FullDataLoaderOptions(loaderOpts)
  }
}

case class TorchTensorDataLoaderOptions(
    batch_size: Int = 1,
    shuffle: Boolean = false,
    sampler: Sampler = null,
    batch_sampler: Sampler = null,
    num_workers: Int = 0,
    max_jobs: Long = 0,
    collate_fn: Any = null,
    pin_memory: Boolean = false,
    drop_last: Boolean = false,
    in_order: Boolean = false,
    timeout: Float = 0,
    worker_init_fn: Any = null,
    prefetch_factor: Option[Int] = Some(2),
    persistent_workers: Boolean = false
) {

  def toNative: org.bytedeco.pytorch.DataLoaderOptions = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(batch_size)
    loaderOpts.batch_size.put(batch_size)
    loaderOpts.drop_last().put(drop_last)
    loaderOpts.enforce_ordering().put(in_order)
    loaderOpts.workers().put(num_workers)
    loaderOpts.max_jobs().put(max_jobs)
//    loaderOpts.timeout(new Milliseconds(new Seconds(timeout.toLong)))
//    loaderOpts.timeout().put(new Milliseconds(timeout.toLong)) //todo Javacpp Bug here timeout will make null pointer, wait for fix
    loaderOpts
  }

  def toNativeFull: org.bytedeco.pytorch.FullDataLoaderOptions = {
    val loaderOpts = new org.bytedeco.pytorch.DataLoaderOptions(batch_size)
    loaderOpts.batch_size.put(batch_size)
    loaderOpts.drop_last().put(drop_last)
    loaderOpts.enforce_ordering().put(in_order)
    loaderOpts.workers().put(num_workers)
    loaderOpts.max_jobs().put(max_jobs)
    //   loaderOpts.timeout(new Milliseconds(new Seconds(timeout.toLong)))
//    loaderOpts.timeout().put(new Milliseconds(timeout.toLong)) //todo Javacpp Bug here timeout will make null pointer, wait for fix
    new org.bytedeco.pytorch.FullDataLoaderOptions(loaderOpts)
  }
}
