package torch.utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  SizeTOptional,
  TensorExampleVectorOptional,
  ChunkTensorBatchDataset as CTBD
}

class ChunkTensorBatchDataset(chunkReader: ChunkTensorDataReader) extends CTBD(chunkReader) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()
}

//  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)
