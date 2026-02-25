package torch
package utils
package tensorboard

//import org.tensorflow.util.event.Event

import scala.collection.mutable
//import scala.collection.mutable.ListBuffer

object Histogram {
  // 生成直方图桶边界
  private def histBuckets(): Seq[Double] = {
    val imin = 1e-12
    val imax = 1e20
    val posBuck = Iterator.iterate(imin)(_ * 1.1).takeWhile(_ < imax).toSeq
    val negBuck = posBuck.map(-_)
    negBuck ++ Seq(0.0) ++ posBuck
  }

  val bucketLimits: Seq[Double] = histBuckets()

  // 计算插入索引
  def insertionIndex(limits: Array[Double], values: Seq[Double]): Seq[Int] = {
    values.map { value =>
      val x = java.util.Arrays.binarySearch(limits, value)
      if (x < 0) -(x + 1) else x
    }
  }

  // 生成直方图数据
  def makeHistogram(values: Seq[Double]): Map[String, Any] = {
    val inIndex = insertionIndex(bucketLimits.toArray, values)
    val kmap = mutable.SortedMap[Int, Int]()
    inIndex.groupBy(identity).foreach { case (k, v) => kmap(k) = v.length }
    val retVals = kmap.values.toSeq.map(_.toDouble)
    val blim = kmap.keys.toSeq.map(bucketLimits)

    Map(
      "min" -> values.min,
      "max" -> values.max,
      "num" -> values.length,
      "sum-squares" -> values.map(x => x * x).sum,
      "sum" -> values.sum,
      "bucket-limit" -> blim,
      "bucket" -> retVals
    )
  }
}
