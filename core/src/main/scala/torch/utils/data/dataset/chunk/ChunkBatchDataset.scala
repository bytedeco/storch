package torch
package utils
package data
package dataset.chunk

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ExampleVectorOptional,
  SizeTOptional,
  ChunkBatchDataset as CBD
}

class ChunkBatchDataset(chunkReader: ChunkDataReader) extends CBD(chunkReader) {

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def offsetAddress[P <: Pointer](i: Long): P = super.offsetAddress(i)
}

object ChunkBatchDataset {}
