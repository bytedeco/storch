package torch
package utils
package tensorboard

import org.tensorflow.framework.summary.Summary
import org.tensorflow.util.event.Event
import org.tensorflow.util.event.Event.What.FileVersion

import java.io.{DataInputStream, FileInputStream}

enum EventType:
  case SUMMARY, FILE_VERSION, SESSION_LOG, LOG_MESSAGE, TAGGED_RUN_METADATA, UNKNOWN

enum EventSummaryType:
  case SCALAR, TENSOR, BLOB_SEQUENCE, IMAGE, HISTOGRAM, UNKNOWN

class SummaryReader(logDir: String, tfEventFilePath: String = "train.tfevents") {

  private val logFilePath = s"$logDir/${tfEventFilePath}"
  private val fileInputStream = new FileInputStream(logFilePath)
  private val dataInputStream = new DataInputStream(fileInputStream)
  private val reader = new TFEventReader(dataInputStream, false)

  def read_event(eventType: String = ""): Unit = readEvent(eventType)

  def readEvent(eventType: String = ""): Unit = {
    try {
      var record: Array[Byte] = reader.read
      while (record != null) {
        val event = Event.parseFrom(record)
        event.what match {
          case Event.What.Summary(summary) =>
            summary.value.foreach { summaryValue =>
              println(summaryValue.value.getClass.getName)
              summaryValue.value match {
                case Summary.Value.Value.Tensor(tensorProto) =>
                  tensorProto.floatVal.headOption.foreach { floatValue =>
                    println(s"Step: ${event.step}, Tag: ${summaryValue.tag}, Value: $floatValue")
                  }
                case sv: Summary.Value.Value.SimpleValue =>
                  // Event(1.752052453297E9,48,Summary(Summary(Vector(Value(,accuracy/train,Some(SummaryMetadata(Some(PluginData(scalars,<ByteString@30b8a058 size=0 contents="">,UnknownFieldSet(Map()))),,,DATA_CLASS_SCALAR,UnknownFieldSet(Map()))),SimpleValue(0.95464104),UnknownFieldSet(Map()))),UnknownFieldSet(Map()))),Some(SourceMetadata(tensorflow.core.util.events_writer,UnknownFieldSet(Map()))),UnknownFieldSet(Map()))
                  // Event(1.7520443742588904E9,999,Summary(Summary(Vector(Value(,Accuracy/train,None,SimpleValue(1.0)
                  println(
                    s"sv Step: ${event.step} event tag ${event} ,summary Tag: ${summaryValue.tag}, Value: ${summaryValue.value.simpleValue.get}"
                  )
                //                  tensorProto.floatVal.headOption.foreach { floatValue =>
                //                    println(s"Step: ${event.step}, Tag: ${summaryValue.tag}, Value: $floatValue")
                //
                case audio: Summary.Value.Value.Audio =>
                  println(s"Step: ${event.step}, Tag: ${summaryValue.tag}, Audio: ${audio}")
                case image: Summary.Value.Value.Image =>
                  println(s"Step: ${event.step}, Tag: ${summaryValue.tag}, Image: ${image}")
                case histogram: Summary.Value.Value.Histo =>
                  println(s"Step: ${event.step}, Tag: ${summaryValue.tag}, Histogram: ${histogram}")
                case _ =>
                  println(
                    s"Unsupported value type for Step ${event.step} tag: ${summaryValue.tag} value ${summaryValue.value.simpleValue.get}"
                  )
              }
            }
          case fv: FileVersion =>
            println(s"FileVersion: ${fv.value}")
          case sl: Event.What.SessionLog =>
            println(s"SessionLog: ${sl.value.status}")
          case _ =>
            println("No summary found in the event")
        }
        record = reader.read
      }
    } catch {
      case e: Exception =>
        println(s"Error reading events: ${e.getMessage}")
    } finally {
      dataInputStream.close()
    }
  }
}
