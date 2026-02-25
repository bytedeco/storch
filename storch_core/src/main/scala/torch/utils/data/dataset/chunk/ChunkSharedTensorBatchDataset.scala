package torch.utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkTensorDataReader,
  SizeTOptional,
  TensorExampleStack,
  TensorExampleVectorOptional,
  ChunkSharedTensorBatchDataset as CSTBD
}
import torch.utils.data.dataset.ChunkTensorDataset

class ChunkSharedTensorBatchDataset(chunkTensorDataset: ChunkTensorDataset)
    extends CSTBD(chunkTensorDataset) {

  override def get_batch(request: Long): TensorExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def multiply(): pytorch.ChunkTensorDataset = super.multiply()

  override def access(): pytorch.ChunkTensorDataset = super.access()

  override def reset(): Unit = super.reset()

  override def map(transform: TensorExampleStack): pytorch.ChunkMapTensorDataset =
    super.map(transform)
}
