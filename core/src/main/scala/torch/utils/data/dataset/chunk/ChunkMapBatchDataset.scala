package torch
package utils.data.dataset.chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ExampleVector,
  SizeTArrayRef,
  SizeTOptional,
  ChunkMapBatchDataset as CMBD
}
import torch.utils.data.dataset.chunk.ChunkMapDataset

class ChunkMapBatchDataset(chunkMapDataset: ChunkMapDataset) extends CMBD(chunkMapDataset) {

  override def get_batch(request: SizeTArrayRef): ExampleVector = super.get_batch(request)

  override def get_batch(request: Long*): ExampleVector = super.get_batch(request: _*)

  override def size(): SizeTOptional = super.size()
}
