package torch.utils.data.stateful

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Example,
  ExampleVector,
  ExampleVectorOptional,
  SizeTOptional,
  AbstractTensor as Tensor,
  JavaStatefulDataset as JSD
}
import torch.utils.data.datareader
import torch.utils.data.datareader.ExampleVectorReader

class StatefulDataset(reader: datareader.ExampleVectorReader) extends JSD(reader) {

  override def get_batch(size: Long) = new ExampleVectorOptional(reader.exampleVec)

  override def size = new SizeTOptional(reader.exampleVec.size)

  override def reset(): Unit = {
//    reader.exampleVec =  new ExampleVector()
  }

  private val ex1 = new Example(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(200.0))
  private val ex2 = new Example(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(400.0))
  private val ex3 = new Example(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(500.0))
  val exampleTestVector = new ExampleVector(ex1, ex2, ex3)

  def getTest(index: Long): Example = {
    exampleTestVector.get(index)
  }
}
