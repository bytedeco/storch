package torch.utils.data.datareader

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.{Tensor, TensorExample, TensorExampleVector}

trait TensorExampleVectorReader(batch: Int = 32) extends Pointer with DataReader {

  def tensorExampleVec: TensorExampleVector // = new TensorExampleVector()

  override def read_chunk(chunk_index: Long): TensorExampleVector = tensorExampleVec

  override def chunk_count: Long = 1

  override def reset(): Unit = {}

  def apply(tensorExampleVector: TensorExampleVector): TensorExampleVector = {
    tensorExampleVector

  }

  def apply(tensorSeq: Seq[Tensor], raw: Boolean = false): TensorExampleVector = {
    val tensorExampleVec = new TensorExampleVector(
      tensorSeq.map(x => new TensorExample(x)).toArray: _*
    )
    tensorExampleVec
  }

  def apply(tensorExampleSeq: Seq[TensorExample]): TensorExampleVector = {
    val tensorExampleVec = new TensorExampleVector(
      tensorExampleSeq.toArray: _*
    )
    tensorExampleVec
  }

}

// new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))

//new TensorExampleVector(
//  new TensorExample(Tensors.create(10.0, 20.0, 50.0, 80.0, 100.0)),
//  new TensorExample(Tensors.create(15.0, 30.0, 50.0, 80.0, 300.0)),
//  new TensorExample(Tensors.create(20.0, 20.0, 50.0, 80.0, 100.0)),
//  new TensorExample(Tensors.create(35.0, 30.0, 50.0, 80.0, 300.0)),
//  new TensorExample(Tensors.create(40.0, 20.0, 50.0, 80.0, 100.0)),
//  new TensorExample(Tensors.create(55.0, 30.0, 50.0, 80.0, 300.0)),
//  new TensorExample(Tensors.create(60.0, 20.0, 50.0, 80.0, 34.0))
//)
