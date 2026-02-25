package torch
package utils
package data
package dataset
package normal

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  Example,
  ExampleVector,
  SizeTArrayRef,
  SizeTOptional,
  JavaDataset as JD
}
import torch.utils.data.datareader.ExampleVectorReader

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class JavaDataset(exampleVectorReader: ExampleVectorReader) extends JD {

  val exampleVector: ExampleVector = exampleVectorReader.exampleVec // .read_chunk(0)

  val ds = new JD() {
    val exampleVector = exampleVectorReader.exampleVec // new ExampleVector(exampleSeq.toArray:_*)

    override def get(index: Long): Example = exampleVector.get(index)

    override def size = new SizeTOptional(exampleVector.size)

  }

  override def get(index: Long): Example = {
    val example = ds.get(index)
    val oldExample = exampleVector.get(index)
    val flag = example.equals(oldExample)
    if (flag) oldExample else example

  } // ds.get(index) //exampleVector.get(index)

  override def size: SizeTOptional = ds.size() // new SizeTOptional(exampleVector.size)

//  def size: Int = length.toInt

  def length = exampleVector.size

  override def get_batch(indices: SizeTArrayRef): ExampleVector =
    super.get_batch(indices) // ds.get_batch(indices) // exampleVector

  def randomSplit(
      lengths: Seq[Int],
      seed: Long = System.currentTimeMillis()
  ): Array[JavaDataset] = {
    require(lengths.sum == length.toInt, "切分长度总和必须等于数据集大小")

    // 创建索引序列并随机打乱 size().get()
    val indices = ArrayBuffer.range(0, length.toInt)
    val random = new Random(seed)
    for (i <- indices.length - 1 to 1 by -1) {
      val j = random.nextInt(i + 1)
      val temp = indices(i)
      indices(i) = indices(j)
      indices(j) = temp
    }

    // split index create dataset 分割索引并创建子数据集
    val splitDatasets = ArrayBuffer[JavaDataset]()
    var current = 0

    for (length <- lengths) {
      val end = current + length
      val subsetIndices = indices.slice(current, end)
      val subsetExampleSeq = subsetIndices.map(idx => ds.get(idx.toLong))
      val exampleVector: ExampleVector = new ExampleVector(subsetExampleSeq.toArray: _*)
      // create sub dataset reader  创建子数据集读取器
      val subsetReader = new ExampleVectorReader {
        override def exampleVec: ExampleVector = ExampleVector(subsetExampleSeq.toArray: _*)
      }

      splitDatasets += new JavaDataset(subsetReader)
      current = end
    }

    splitDatasets.toArray
  }

  /** 按比例随机切分数据集
    *
    * @param ratios
    *   每个子数据集的比例
    * @param seed
    *   随机种子
    * @return
    *   切分后的子数据集数组
    */
  def randomSplitByRatio(
      ratios: Seq[Double],
      seed: Long = System.currentTimeMillis()
  ): Array[JavaDataset] = {
    require(math.abs(ratios.sum - 1.0) < 1e-6, "比例总和必须接近1.0")

    val totalSize = length.toInt
    val lengths = ratios.map(ratio => (ratio * totalSize).toInt)

    // 处理四舍五入误差
    val remaining = totalSize - lengths.sum
    if (remaining > 0) {
      lengths.updated(lengths.length - 1, lengths.last + remaining)
    }

    randomSplit(lengths, seed)
  }

}

//  override def get(index: Long): Example = super.get(index) // 失败

//    {
//    val batch = 32 //exampleVectorReader.batch
//    val exampleVector = new ExampleVector()
//    for (i <- 0 until batch) {
//       exampleVector.put( exampleVectorReader.read_chunk(i.toLong).get*)
//    }
//    exampleVector
//  }

//     val exampleVector = new ExampleVector(exampleSeq.toArray:_*)
//    override def get(index: Long): Example = exampleVector.get(index)
//
//    override def size = new SizeTOptional(exampleVector.size)
//

// def random_split[T](dataset: JavaDataset[T]
//  def random_split(dataset: JavaDataset,
//                      lengths: Seq[Double | Int],
//                      generator: Option[Generator] = None): Seq[JavaDataset] = {
//    // 处理长度参数，支持小数比例和整数长度
//    val subsetLengths: Seq[Int] = {
//      if (math.abs(lengths.sum - 1.0) < 1e-6 && lengths.sum <= 1.0) {
//        // 如果是小数比例，计算实际长度
//        val baseLengths = lengths.map {
//          case frac: Double =>
//            if (frac < 0 || frac > 1) {
//              throw new IllegalArgumentException(s"比例值必须在0到1之间: $frac")
//            }
//            math.floor(dataset.size * frac).toInt
//          case int: Int =>
//            if (int < 0) {
//              throw new IllegalArgumentException(s"长度不能为负数: $int")
//            }
//            int
//        }
//        // 处理余数，使用轮询方式分配剩余样本
//        val remainder = dataset.size - baseLengths.sum
//        val result = baseLengths.toBuffer
//        for (i <- 0 until remainder) {
//          val idx = i % result.length
//          result(idx) = result(idx) + 1
//        }
//
//        result.toSeq
//      } else {
//        // 否则直接使用整数长度
//        val intLengths = lengths.map {
//          case int: Int =>
//            if (int < 0) throw new IllegalArgumentException(s"长度不能为负数: $int")
//            int
//          case frac: Double =>
//            throw new IllegalArgumentException(s"长度总和不等于1.0，但包含小数: $frac")
//        }
//        if (intLengths.sum != dataset.size) {
//          throw new IllegalArgumentException(s"整数长度总和必须等于数据集大小! 总和: ${intLengths.sum}, 数据集大小: ${dataset.size}")
//        }
//        intLengths
//      }
//      // 生成随机排列的索引
//      val randPerm = {
//        val perm = generator match {
//          case Some(gen) => torch.randperm(dataset.size.get(), gen)
//          case None => torch.randperm(dataset.size.get())
//        }
//        // 转换为Scala的Int序列
//        val indices = new Array[Int](dataset.size)
//        for (i <- 0 until dataset.size) {
//          indices(i) = perm.item(i).toInt
//        }
//        indices.toSeq
//      }
//
//      // 创建子集
//      val subsets = ListBuffer[JavaDataset]()
//      var offset = 0
//
//      for (length <- subsetLengths) {
//        val subsetIndices = randPerm.slice(offset, offset + length)
//        subsets += JavaDataset(dataset, subsetIndices)
//        offset += length
//      }
//
//      subsets.toSeq
//    }
//
//
//}
