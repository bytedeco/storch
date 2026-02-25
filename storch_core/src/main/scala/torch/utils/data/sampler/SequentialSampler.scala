package torch.utils.data.sampler

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  InputArchive,
  OutputArchive,
  SizeTOptional,
  SizeTVectorOptional,
  SequentialSampler as SS
}

class SequentialSampler(size: Long) extends SS(size) with Sampler {

  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)

  override def reset(): Unit = super.reset()

  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

  override def index(): Long = super.index()
}
