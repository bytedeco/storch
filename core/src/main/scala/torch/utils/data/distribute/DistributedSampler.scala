package torch
package utils
package data
package distribute

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  DistributedSampler as DS
}
import torch.internal.NativeConverters.{toNative}
import torch.utils.data.sampler.Sampler

class DistributedSampler(epoch: Int) extends DS(epoch.toNative) with Sampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

}
