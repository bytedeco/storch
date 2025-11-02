package torch.utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  InputArchive,
  OutputArchive,
  SizeTOptional,
  TensorExampleVectorOptional,
  ChunkStatefulTensorDataset as CSTD
}

class ChunkStatefulTensorDataset(chunkReader: ChunkTensorDataReader) extends CSTD(chunkReader) {

  override def reset(): Unit = super.reset()

  override def save(archive: OutputArchive): Unit = super.save(archive)

  override def load(archive: InputArchive): Unit = super.load(archive)

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
