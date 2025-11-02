package torch.utils.data.datareader

import org.bytedeco.pytorch.{Example, ExampleVector, Tensor, ChunkDataReader as CDR}

class ChunkDataReader(batch: Int = 32) extends CDR with DataReader {

  var exampleVec: ExampleVector =
    new ExampleVector()

  override def read_chunk(chunk_index: Long) = exampleVec

  override def chunk_count: Long = 1

  override def reset(): Unit = {}

  def apply(exampleVector: ExampleVector): ExampleVector = {
    this.exampleVec = exampleVector
    this.exampleVec
  }

  def apply(tensorSeq: Seq[(Tensor, Tensor)]): ExampleVector = {

    this.exampleVec = new ExampleVector(tensorSeq.map(x => new Example(x._1, x._2)).toArray: _*)
    this.exampleVec
  }

}

// new Example(Tensors.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(200.0)),
// new Example(Tensors.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(400.0)),
// new Example(Tensors.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(500.0)),
// new Example(Tensors.create(35.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(600.0)),
// new Example(Tensors.create(40.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(700.0)),
// new Example(Tensors.create(55.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(800.0)),
// new Example(Tensors.create(60.0, 20.0, 50.0, 80.0, 100.0), Tensors.create(900.0)),
// new Example(Tensors.create(75.0, 30.0, 50.0, 80.0, 300.0), Tensors.create(300.0)))

//  def batchSplit(exampleVector: ExampleVector): ExampleVector = {
//    val batch = exampleVector.begin()
//    var it = exampleVector.begin
//    var batchIndex = 0
//    var tensorSize = 0
//    val tensorBuffer = new ListBuffer[Tensor]()
//    while (!it.equals(exampleVector.end)) {
//      val batch = it.access
//      tensorSize += batch.size()
//      tensorBuffer.+= batch
//        it.increment()
//    }
//    tensorBuffer.foldLeft(tensorBuffer(0))(_.concat _)
//    val batchedExampleVector = new ExampleVector()
//    for (i <- 0 until batch) {
//      val start = i * batch
//      val end = start + batch
//      val batchedExample = new ExampleVector()
//      for (j <- start until end) {
//        batchedExample.add(exampleVector.get(j))
//      }
//      batchedExampleVector.add(batchedExample)
//    }
//    batchedExampleVector
//  }
