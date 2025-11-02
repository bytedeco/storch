package torch
package utils
package data
package stream

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{SizeTOptional, TensorExampleVector, JavaStreamTensorDataset as STD}
import torch.utils.data.datareader.TensorExampleVectorReader

class StreamTensorDataset(reader: datareader.TensorExampleVectorReader) extends STD {

  override def get_batch(request: Long): TensorExampleVector = reader.tensorExampleVec

  override def size(): SizeTOptional = new SizeTOptional(reader.tensorExampleVec.size)
}
