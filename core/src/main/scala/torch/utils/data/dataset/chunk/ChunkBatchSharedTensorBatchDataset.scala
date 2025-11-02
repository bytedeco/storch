package torch
package utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{ChunkTensorDataReader, ChunkBatchSharedTensorBatchDataset as CBSTBD}

class ChunkBatchSharedTensorBatchDataset(chunkReader: ChunkTensorDataReader)
    extends CBSTBD(chunkReader) {

  override def get_batch(request: Long): _root_.org.bytedeco.pytorch.TensorExampleVectorOptional =
    super.get_batch(request)

  override def size(): _root_.org.bytedeco.pytorch.SizeTOptional = super.size()

  override def map(
      transform: _root_.org.bytedeco.pytorch.TensorExampleStack
  ): _root_.org.bytedeco.pytorch.ChunkMapTensorDataset = super.map(transform)
}
