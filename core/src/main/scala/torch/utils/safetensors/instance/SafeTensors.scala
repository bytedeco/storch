package torch.utils.safetensors.instance

//import java.util.Locale
import scala.collection.mutable
import scala.collection.mutable.{ListBuffer, Map}

object SafeTensors {
  enum DataType:
    case F64, F32, F16, BF16, I64, I32, I16, I8, BOOL

  case class HeaderElement(
      name: String,
      dataType: SafeTensors.DataType,
      shape: Seq[Long],
      offsets: Seq[Long]
  )

}
class SafeTensors {

  var sizeOfHeader: Long = 0L
  var header: ListBuffer[SafeTensors.HeaderElement] = null
  var metadata: mutable.Map[String, String] = null
  var filename: String = null

  var name: String = null
  var dataType: SafeTensors.DataType = null
  var shape: Seq[Long] = null
  var offsets: Seq[Long] = null
}
