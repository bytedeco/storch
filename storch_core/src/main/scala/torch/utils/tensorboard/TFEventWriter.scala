package torch
package utils
package tensorboard

import java.io.*
import java.nio.{ByteBuffer, ByteOrder}

class TFEventWriter(private val output: DataOutput) {
  @throws[IOException]
  def write(event: Array[Byte]): Unit = {

    /** TFEvent format: uint64 length uint32 masked_crc32_of_length byte data[length] uint32
      * masked_crc32_of_data
      */
    val len = toInt64LE(event.length)
    output.write(len)
    output.write(toInt32LE(Crc32C.maskedCrc32c(len)))
    output.write(event)
    output.write(toInt32LE(Crc32C.maskedCrc32c(event)))
//    output.write(toFloatLE(Crc32C.maskedCrc32c(event)))
  }

  private def toInt64LE(data: Long) = {
    val buff = new Array[Byte](8)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putLong(data)
    buff
  }

  private def toInt32LE(data: Int) = {
    val buff = new Array[Byte](4)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putInt(data)
    buff
  }

  private def toInt16LE(data: Short) = {
    val buff = new Array[Byte](2)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putShort(data)
    buff
  }

  private def toFloatLE(data: Float) = {
    val buff = new Array[Byte](4)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putFloat(data)
    buff
  }

  private def toDoubleLE(data: Double) = {
    val buff = new Array[Byte](8)
    val bb = ByteBuffer.wrap(buff)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb.putDouble(data)
    buff
  }
}
