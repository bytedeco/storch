package torch.utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ExampleVectorOptional,
  SizeTOptional,
  ChunkStatefulDataset as CSD
}

class ChunkStatefulDataset(chunkReader: ChunkDataReader) extends CSD(chunkReader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}
