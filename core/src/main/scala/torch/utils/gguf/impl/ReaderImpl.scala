package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*
import torch.utils.gguf.{GGMLType, GGUF, MetadataValueType, TensorInfo}

import java.io.{EOFException, IOException}
import java.nio.channels.ReadableByteChannel
import java.nio.charset.StandardCharsets
import java.nio.{Buffer, ByteBuffer, ByteOrder}
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object ReaderImpl {
  private[impl] val GGUF_MAGIC = 0x46554747
  private[impl] val ALIGNMENT_DEFAULT_VALUE = 32 // must be a power of 2
  private[impl] val ALIGNMENT_KEY = "general.alignment"
  private val SUPPORTED_GGUF_VERSIONS = List(2, 3)
}

final class ReaderImpl {
  private var version = 0
  private var alignment = 0
  private var metadata: mutable.Map[String, Any] = null
  private var metadataTypes: mutable.Map[String, MetadataValueType] = null
  private var totalBytesRead = 0L
  final private val BB_8 = ByteBuffer.allocate(java.lang.Long.BYTES).order(ByteOrder.nativeOrder)

  @throws[IOException]
  private[impl] def readImpl(byteChannel: ReadableByteChannel) = {
    // The header of the file.
    val tensorCount = readHeader(byteChannel) // gguf_header_t header;
    // Tensor infos, which can be used to locate the tensor data.
    // gguf_tensor_info_t tensor_infos[header.tensor_count];
    val tensorInfos = new mutable.LinkedHashMap[String, TensorInfo] // (tensorCount)
    for (i <- 0 until tensorCount) {
      val ti = readTensorInfo(byteChannel)
      assert(!tensorInfos.contains(ti.name))
      tensorInfos.put(ti.name, ti)
    }
    // Padding to the nearest multiple of `ALIGNMENT`.
    // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
    // long _padding = -byteChannel.position() & (ALIGNMENT - 1);
    val padding = GGUFImpl.padding(totalBytesRead, getAlignment).toInt
    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be
    // close
    // or identical to the data in the original model file, but may be different due to quantization
    // or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or
    // as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos`
    // entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between
    // tensors
    // should be padded to `ALIGNMENT` bytes.
    // uint8_t tensor_data[];
    val tensorDataOffset = totalBytesRead + padding
    new GGUFImpl(this.version, tensorDataOffset, this.metadata, this.metadataTypes, tensorInfos)
  }

  @throws[IOException]
  private def readGGMLType(byteChannel: ReadableByteChannel) = {
    val ggmlTypeId = readInt(byteChannel) // ggml_type type;
    GGMLType.fromId(ggmlTypeId)
  }

  @throws[IOException]
  private def readTensorInfo(byteChannel: ReadableByteChannel) = {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    val name = readString(byteChannel) // gguf_string_t name;
    assert(name.length <= 64)
    // The number of shape in the tensor.
    // Currently at most 4, but this may change in the future.
    val n_dimensions = readInt(byteChannel) // uint32_t n_dimensions;
    assert(n_dimensions <= 4)
    // The shape of the tensor.
    val dimensions = new Array[Long](n_dimensions) // uint64_t shape[n_dimensions];
    for (i <- 0 until n_dimensions) {
      dimensions(i) = readLong(byteChannel)
    }
    // The type of the tensor.
    val ggmlType = readGGMLType(byteChannel) // ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    // Must be a multiple of `ALIGNMENT`.
    val offset = readLong(byteChannel) // uint64_t offset;
    assert(offset % getAlignment == 0)
    TensorInfo.create(name, dimensions, ggmlType, offset)
  }

  @throws[IOException]
  private def readString(byteChannel: ReadableByteChannel) = {
    // A string in GGUF.
    // The length of the string, in bytes.
    val len = Math.toIntExact(readLong(byteChannel)) // uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    val bytes = new Array[Byte](len) // char string[len];
    readBytes(byteChannel, bytes)
    new String(bytes, StandardCharsets.UTF_8)
  }

  @throws[IOException]
  private def readHeader(byteChannel: ReadableByteChannel) = {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    val magic = readInt(byteChannel) //    uint32_t magic;
    if (magic != ReaderImpl.GGUF_MAGIC)
      throw new IllegalArgumentException(
        "Invalid header.magic: " + magic + " expected: " + ReaderImpl.GGUF_MAGIC
      )
    // The version of the format implemented.
    // Must be `3` for version described in this spec.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    this.version = readInt(byteChannel) // uint32_t version;
    if (!ReaderImpl.SUPPORTED_GGUF_VERSIONS.contains(version))
      throw new IllegalArgumentException(
        "Unsupported header.version:" + version + " expected: " + ReaderImpl.SUPPORTED_GGUF_VERSIONS
      )
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    val tensorCount = Math.toIntExact(readLong(byteChannel)) // uint64_t tensor_count;
    // The number of metadata key-value pairs.
    val metadataKeyValueCount =
      Math.toIntExact(readLong(byteChannel)) // uint64_t metadata_kv_count;

    // The metadata key-value pairs.
    // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
    this.metadata = new mutable.LinkedHashMap[String, Any] // (metadataKeyValueCount)
    this.metadataTypes =
      new mutable.LinkedHashMap[String, MetadataValueType] // (metadataKeyValueCount)
    for (i <- 0 until metadataKeyValueCount) {
      // The key of the metadata. It is a standard GGUF string, with the following caveats:
      // - It must be a valid ASCII string.
      // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by
      // a `.`.
      // - It must be at most 2^16-1/65535 bytes long.
      // Any keys that do not follow these rules are invalid.
      val key = readString(byteChannel) // gguf_string_t key;
      assert(key.length < (1 << 16))
      assert(
        key.codePoints.allMatch((cp: Int) =>
          ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.'
        )
      )
      // The type of the value.
      // Must be one of the `gguf_metadata_value_type` values.
      val valueType = readMetadataValueType(byteChannel) // gguf_metadata_value_type value_type;
      // The value.
      val value = readMetadataValueOfType(byteChannel, key, valueType)
      assert(!metadata.contains(key))
      assert(!metadataTypes.contains(key))
      metadata.put(key, value)
      metadataTypes.put(key, valueType)
    }
    tensorCount
  }

  @throws[IOException]
  private def readArray(byteChannel: ReadableByteChannel, key: String) = {
    // Any value type is valid, including arrays.
    val componentType = readMetadataValueType(byteChannel) // gguf_metadata_value_type type;

    // Record the component type.
    this.metadataTypes.put(key + "[]", componentType)
    // Number of elements, not bytes.
    val len = Math.toIntExact(readLong(byteChannel)) // uint64_t len;

    // The array of values.
    // gguf_metadata_value_t array[len];
    componentType match {
//      case UINT8 =>
      case INT8 | UINT8 =>
        val bytes = new Array[Byte](len)
        for (i <- 0 until len) {
          bytes(i) = readByte(byteChannel)
        }
        bytes
//      case UINT16 =>
      case INT16 | UINT16 =>
        val shorts = new Array[Short](len)
        for (i <- 0 until len) {
          shorts(i) = readShort(byteChannel)
        }
        shorts
//      case UINT32 =>
      case INT32 | UINT32 =>
        val ints = new Array[Int](len)
        for (i <- 0 until len) {
          ints(i) = readInt(byteChannel)
        }
        ints
      case UINT64 =>
      case INT64 =>
        val longs = new Array[Long](len)
        for (i <- 0 until len) {
          longs(i) = readLong(byteChannel)
        }
        longs
      case FLOAT32 =>
        val floats = new Array[Float](len)
        for (i <- 0 until len) {
          floats(i) = readFloat(byteChannel)
        }
        floats
      case FLOAT64 =>
        val doubles = new Array[Double](len)
        for (i <- 0 until len) {
          doubles(i) = readDouble(byteChannel)
        }
        doubles
      case BOOL =>
        val booleans = new Array[Boolean](len)
        for (i <- 0 until len) {
          booleans(i) = readBoolean(byteChannel)
        }
        booleans
      case STRING =>
        val strings = new Array[String](len)
        for (i <- 0 until len) {
          strings(i) = readString(byteChannel)
        }
        strings
      case ARRAY =>
        throw new UnsupportedOperationException("Cannot read array of arrays")
      case _ =>
        throw new UnsupportedOperationException("Found array of unknown type " + componentType)
    }
  }

  @throws[IOException]
  private def readMetadataValueOfType(
      byteChannel: ReadableByteChannel,
      key: String,
      valueType: MetadataValueType
  ) = valueType match {
//    case UINT8 => // fall-through
    case INT8 | UINT8 =>
      readByte(byteChannel)
//    case UINT16 => // fall-through
    case INT16 | UINT16 =>
      readShort(byteChannel)
//    case UINT32 => // fall-through
    case INT32 | UINT32 =>
      readInt(byteChannel)
    case FLOAT32 =>
      readFloat(byteChannel)
//    case UINT64 => // fall-through
    case INT64 | UINT64 =>
      readLong(byteChannel)
    case FLOAT64 =>
      readDouble(byteChannel)
    case BOOL =>
      readBoolean(byteChannel)
    case STRING =>
      readString(byteChannel)
    case ARRAY =>
      readArray(byteChannel, key)
    case _ =>
      throw new IllegalArgumentException
  }

  @throws[IOException]
  private def readFully(byteChannel: ReadableByteChannel, byteBuffer: ByteBuffer) = {
    while (byteBuffer.position < byteBuffer.limit) {
      val bytesRead = byteChannel.read(byteBuffer)
      if (bytesRead < 0) throw new EOFException
      else if (bytesRead > 0) totalBytesRead += bytesRead
    }
    byteBuffer
  }

  @throws[IOException]
  private def readBytes(byteChannel: ReadableByteChannel, bytes: Array[Byte]): Unit = {
    readFully(byteChannel, ByteBuffer.wrap(bytes))
  }

  @throws[IOException]
  private def readByte(byteChannel: ReadableByteChannel) =
    readFully(byteChannel, BB_8.clear.limit(1).asInstanceOf[ByteBuffer]).get(0)

  @throws[IOException]
  private def readBoolean(byteChannel: ReadableByteChannel) = readByte(byteChannel) != 0

  @throws[IOException]
  private def readShort(byteChannel: ReadableByteChannel) =
    readFully(byteChannel, BB_8.clear.limit(2).asInstanceOf[ByteBuffer]).getShort(0)

  @throws[IOException]
  private def readInt(byteChannel: ReadableByteChannel) =
    readFully(byteChannel, BB_8.clear.limit(4).asInstanceOf[ByteBuffer]).getInt(0)

  @throws[IOException]
  private def readLong(byteChannel: ReadableByteChannel) =
    readFully(byteChannel, BB_8.clear.limit(8).asInstanceOf[ByteBuffer]).getLong(0)

  @throws[IOException]
  private def readFloat(byteChannel: ReadableByteChannel) =
    java.lang.Float.intBitsToFloat(readInt(byteChannel))

  @throws[IOException]
  private def readDouble(byteChannel: ReadableByteChannel) =
    java.lang.Double.longBitsToDouble(readLong(byteChannel))

  @throws[IOException]
  private def readMetadataValueType(byteChannel: ReadableByteChannel) = {
    val index = readInt(byteChannel)
    MetadataValueType.fromIndex(index)
  }

  private def getAlignment: Int = {
    if (alignment != 0) return alignment
    assert(
      !metadataTypes.contains(ReaderImpl.ALIGNMENT_KEY) || (metadataTypes.get(
        ReaderImpl.ALIGNMENT_KEY
      ) eq MetadataValueType.UINT32)
    )
    alignment = metadata
      .getOrElse(ReaderImpl.ALIGNMENT_KEY, ReaderImpl.ALIGNMENT_DEFAULT_VALUE)
      .asInstanceOf[Int]
    assert(Integer.bitCount(alignment) == 1, "alignment must be a power of two")
    assert(alignment > 0)
    alignment
  }
}
