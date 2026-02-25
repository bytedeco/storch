package torch.utils.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  BatchSizeOptional,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  BatchSizeSampler as BSS
}

trait BatchSizeSampler extends BSS {
  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def next(batch_size: Long): BatchSizeOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)
}
