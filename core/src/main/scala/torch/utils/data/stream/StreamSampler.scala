package torch
package utils
package data
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BatchSizeOptional,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  StreamSampler as SS
}
import torch.utils.data.sampler.BatchSizeSampler

class StreamSampler(epoch_size: Long) extends SS(epoch_size) with BatchSizeSampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def reset(): Unit = super.reset()

  override def next(batch_size: Long): BatchSizeOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
