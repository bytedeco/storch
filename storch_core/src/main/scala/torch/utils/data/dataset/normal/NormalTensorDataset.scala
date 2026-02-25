package torch.utils.data.dataset.normal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  SizeTArrayRef,
  SizeTOptional,
  TensorExample,
  TensorExampleVector,
  JavaTensorDataset as TD
}
import torch.utils.data.datareader
import torch.utils.data.datareader.TensorExampleVectorReader

class NormalTensorDataset(reader: datareader.TensorExampleVectorReader) extends TD {

  var tensorExampleVec: TensorExampleVector = reader.read_chunk(0)

  val ds = new TD() {
    val tex = reader.read_chunk(
      0
    )
    override def get(index: Long): TensorExample = {
      tex.get(index)
      //                    return super.get(index);
    }

    override def get_batch(indices: SizeTArrayRef): TensorExampleVector =
      tex // .get_batch(indices) // ds.get_batch(indices) // exampleVector
    override def size = new SizeTOptional(tex.size)
  }
  override def get(index: Long): TensorExample =
    ds.get(index)

  override def size = new SizeTOptional(tensorExampleVec.size)

  def length = tensorExampleVec.size

  override def position(position: Long): TD = super.position(position)

  override def getPointer(i: Long): TD = super.getPointer(i)

  override def get_batch(indices: SizeTArrayRef): TensorExampleVector =
    ds.get_batch(indices)
}

//  override def get_batch(request: Long): TensorExampleVector = super.get_batch(request)

// new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))
//exampleVector.get(index) {
//    tensorExampleVec.get(index)
