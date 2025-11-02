package torch.utils.models.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ListBuffer

type IntOrString = Int | String

//| ComplexNN
//todo  has init error, cause by SENet
final class EmbeddingModule[ParamType <: FloatNN: Default](
    datatypes: Seq[Map[String, IntOrString]],
    useSeNet: Boolean
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  // todo  has init error ,sparseNum must >0
  var sparseNum = datatypes.length
  var denseDim = 0
  var sparseDim = 0
//  var sparseNum = 0
  val embNets = nn.ModuleList[ParamType]()
//  val seNet = if (useSeNet) register(new SENet(sparseNum)) else None

  val seNet = register(new SENet(sparseNum))

  var index = 1
  for (datatype <- datatypes) {
    datatype("type") match {
      case "SparseEncoder" | "BucketSparseEncoder" =>
        val length: Int = datatype("length") match {
          case i: Int    => i
          case s: String => 0
        }
        val emb_dim: Int = datatype("emb_dim") match {
          case i: Int    => i
          case s: String => 0
        }
        val emb = nn.Embedding(length, emb_dim)
        embNets.insert(index, emb.asInstanceOf[TensorModule[ParamType]])
        sparseDim += emb_dim
        sparseNum += 1
      case "MultiSparseEncoder" =>
        val length = datatype("length") match {
          case i: Int    => i
          case s: String => 0
        }
        val emb_dim = datatype("emb_dim") match {
          case i: Int    => i
          case s: String => 0
        }
        val emb_bag = nn.EmbeddingBag(length, emb_dim, mode = "sum")
//        embNets.append(emb_bag.asInstanceOf[TensorModule[ParamType]])
        embNets.insert(index, emb_bag.asInstanceOf[TensorModule[ParamType]])
        sparseDim += emb_dim
        sparseNum += 1
      case "DenseEncoder" =>
        denseDim += 1
        denseNum += 1
      case "VecDenseEncoder" =>
        val size: Int = datatype("size") match {
          case i: Int    => i
          case s: String => 0
        }
        denseDim += size
        denseNum += 1
    }
    index += 1
  }
  var denseNum = 0

  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
    val embOutput = ListBuffer[Tensor[ParamType]]()
    val denseOutput = ListBuffer[Tensor[ParamType]]()
    var embIndex = 0
    val seNetInput = ListBuffer[Tensor[ParamType]]()

    for (index <- datatypes.indices) {
      val datatype = datatypes(index)
      datatype("type") match {
        case "MultiSparseEncoder" =>
          val start = datatype("index") match {
            case i: Int    => i
            case s: String => 0
          }
          val size = datatype("size") match {
            case i: Int    => i
            case s: String => 0
          }
          val end = start + size
          val vec = embNets(embIndex)(x.index(::, start.&&(end)))
          if (useSeNet) {
            seNetInput.append(vec.mean(1).view(-1, 1))
          }
          embOutput.append(vec)
          embIndex += 1
        case "SparseEncoder" | "BucketSparseEncoder" =>
          val dataIndex = datatype("index") match {
            case i: Int    => i
            case s: String => 0
          }
          val vec = embNets(embIndex)(x.index(::, dataIndex))
          embOutput.append(vec)
          if (useSeNet) {
            seNetInput.append(vec.mean(1).view(-1, 1))
          }
          embIndex += 1
        case "DenseEncoder" | "VecDenseEncoder" =>
          val start = datatype("index") match {
            case i: Int    => i
            case s: String => 0
          }
          val size = datatype("size") match {
            case i: Int    => i
            case s: String => 0
          }
          val end = start + size
          //          val end = start + datatype("size")
          val vec = x.index(::, start.&&(end))
          denseOutput.append(vec)
      }
    }

    if (seNetInput.nonEmpty && useSeNet) {
      val input = torch.cat(seNetInput.toSeq, 1).to(dtype = this.paramType)
      val seNetOutput = seNet(input)
      for (i <- 0 until sparseNum) {
        embOutput(i) = embOutput(i) * seNetOutput.index(-1, i.&&(i + 1))
      }
    }

    val output = (denseOutput, embOutput) match {
      case (dense, emb) if dense.nonEmpty && emb.nonEmpty =>
        torch.cat(Seq(torch.cat(emb.toSeq, 1), torch.cat(dense.toSeq, 1)), 1) // .toFloat()
      case (dense, emb) if dense.nonEmpty =>
        torch.cat(dense.toSeq, 1)
      case (dense, emb) if emb.nonEmpty =>
        torch.cat(emb.toSeq, 1)
      case _ =>
        torch.empty().to(dtype = this.paramType)
      //        Tensor.empty()
    }

    output.to(dtype = this.paramType)
  }
}
