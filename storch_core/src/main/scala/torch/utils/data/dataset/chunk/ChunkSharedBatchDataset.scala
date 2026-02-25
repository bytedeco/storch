package torch
package utils
package data
package dataset
package chunk

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  ChunkDataReader,
  ExampleStack,
  ExampleVectorOptional,
  SizeTOptional,
  ChunkMapDataset as CMD,
  ChunkSharedBatchDataset as CSBD
}

class ChunkSharedBatchDataset(chunkDataset: ChunkDataset) extends CSBD(chunkDataset) {

  var native: CMD = null

  override def get_batch(request: Long): ExampleVectorOptional = super.get_batch(request)

  override def size(): SizeTOptional = super.size()

  override def map(transform: ExampleStack): CMD = {

    native = super.map(transform)
//    native.transform()
//    native.asInstanceOf[ChunkMapDataset]
//    new ChunkMapDataset()
    native

  }
}
