package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*
import torch.utils.gguf.{Builder, GGUF, MetadataValueType, TensorInfo}
import torch.utils.gguf.*

import java.lang.reflect.Array as JArray
import java.nio.charset.StandardCharsets
import java.util.function.Function
import java.util.stream.Collectors
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object BuilderImpl {
  private val DEFAULT_VERSION = 3

  private[impl] def fromExisting(gguf: GGUF) = new BuilderImpl()
    .setVersion(gguf.getVersion)
    .setMetadata(BuilderImpl.reconstructMetadata(gguf).toMap)
    .setMetadataTypes(BuilderImpl.reconstructTypes(gguf).toMap)
    .setTensors(BuilderImpl.fromCollection(gguf.getTensors))

//  private def reconstructMetadata(gguf: GGUF) = gguf.getMetadataKeys.stream.collect(Collectors.toMap(Function.identity, (key: String) => gguf.getValue(classOf[AnyRef], key), (a: AnyRef, b: AnyRef) => a, util.LinkedHashMap.`new`))

  private def reconstructMetadata(gguf: GGUF): mutable.LinkedHashMap[String, Any] =
    val result = mutable.LinkedHashMap[String, Any]()
    gguf.getMetadataKeys.foreach { key =>
      result(key) = gguf.getValue[Any](classOf[Any], key)
    }
    result
  private def reconstructTypes(gguf: GGUF) = {
    val metadataTypes = new mutable.HashMap[String, MetadataValueType]
    for (key <- gguf.getMetadataKeys) {
      val valueType = gguf.getType(key)
      assert(valueType != null)
      metadataTypes.put(key, valueType)
      if (valueType eq MetadataValueType.ARRAY)
        metadataTypes.put(key + "[]", gguf.getComponentType(key))
    }
    metadataTypes
  }

//  private def fromCollection(tensors: Seq[TensorInfo]) = tensors.collect(Collectors.toMap(TensorInfo.name, Function.identity, (a: TensorInfo, b: TensorInfo) => {
//    throw new IllegalArgumentException("duplicated tensor names")
//  }, util.LinkedHashMap.`new`))

  private def fromCollection(
      tensors: Iterable[TensorInfo]
  ): mutable.LinkedHashMap[String, TensorInfo] =
    val result = mutable.LinkedHashMap[String, TensorInfo]()
    tensors.foreach { tensor =>
      if result.contains(tensor.name) then
        throw new IllegalArgumentException("duplicated tensor names")
      result(tensor.name) = tensor
    }
    result
  private def sizeOfStringValue(value: String) =
    java.lang.Long.BYTES // uint64_t len + value.getBytes(StandardCharsets.UTF_8).length.toLong

  private def sizeOfTensorInfo(tensorInfo: TensorInfo) = sizeOfStringValue(
    tensorInfo.name
  ) // gguf_string_t name + Integer.BYTES// uint32_t n_dimensions; + Long.BYTES * tensorInfo.shape.length.toLong// uint64_t dimensions[n_dimensions]; + Integer.BYTES// ggmlType type + Long.BYTES// uint64_t offset

  @SuppressWarnings(Array("unchecked"))
  private def castTo[T](key: String, value: Any, targetClass: Class[? <: T]) = {
//    Objects.requireNonNull(value)
    try
      if (targetClass.isPrimitive) GGUFImpl.toBoxedClass(targetClass).cast(value).asInstanceOf[T]
      else targetClass.cast(value)
    catch {
      case e: ClassCastException =>
        throw new IllegalArgumentException(
          "Expected value type " + targetClass + " but got " + value.getClass + " for key '" + key + "'"
        )
    }
  }
}

final class BuilderImpl extends AbstractBuilder {
  private var version = BuilderImpl.DEFAULT_VERSION
  private var metadata = new mutable.LinkedHashMap[String, Any]
  private var metadataTypes = new mutable.LinkedHashMap[String, MetadataValueType]
  private var tensorInfos = new mutable.LinkedHashMap[String, TensorInfo]

  override def setVersion(newVersion: Int): BuilderImpl = {
    this.version = newVersion
    this
  }

  // newMetadata: Map[String, AnyRef]
  private[impl] def setMetadata(newMetadata: Map[String, Any]) = {
    // Must preserve insertion order.
//    assert(newMetadata.isInstanceOf[util.LinkedHashMap[_, _]])
    this.metadata.clear()
    this.metadata ++= newMetadata
    this
  }

  private[impl] def setMetadataTypes(newMetadataTypes: Map[String, MetadataValueType]) = {
    this.metadataTypes.clear()
    this.metadataTypes ++= newMetadataTypes
    this
  }

  override def build(recomputeTensorOffsets: Boolean): GGUF = {
//    assert(this.metadata.keySet(key: String) => this.metadataTypes.contains(key)))
    val freshTensorDataOffset = computeTensorDataOffset
    val freshTensorInfos =
      if (recomputeTensorOffsets) computeTensorOffsets
      else this.tensorInfos
    new GGUFImpl(
      this.version,
      freshTensorDataOffset,
      this.metadata,
      this.metadataTypes,
      freshTensorInfos
    )
  }

  override def clone: BuilderImpl = new BuilderImpl()
    .setVersion(getVersion)
    .setMetadata(this.metadata.toMap)
    .setMetadataTypes(
      this.metadataTypes.toMap
    ) // new mutable.LinkedHashMap[String, MetadataValueType](this.metadataTypes))
    .setTensors(
      this.tensorInfos
    ) // new mutable.LinkedHashMap[String, TensorInfo](this.tensorInfos))

  override def getVersion: Int = this.version

  private def sizeOfTaggedValue(key: String, value: Any) = {
    val valueType = this.metadataTypes.getOrElse(key, null)
//    Objects.requireNonNull(valueType)
    var totalSize = Integer.BYTES // gguf_metadata_value_type: uint32_t type;
    valueType match {
      case UINT8 | INT8 | UINT16 | INT16 | UINT32 | INT32 | FLOAT32 | BOOL | UINT64 | INT64 |
          FLOAT64 => // fall-through
        totalSize += valueType.byteSize()
      case STRING =>
        totalSize += BuilderImpl.sizeOfStringValue(value.asInstanceOf[String])
      case ARRAY =>
        totalSize += sizeOfArray(key, value).toInt
    }
    totalSize
  }

  private def sizeOfArray(key: String, arrayValue: Any): Long = {
    assert(arrayValue.getClass.isArray)
    val componentType = this.metadataTypes.get(key + "[]").get
    if (componentType eq MetadataValueType.ARRAY)
      throw new IllegalArgumentException("array of arrays not supported for key '" + key + "'")
    if (componentType eq MetadataValueType.STRING) {
      val stringArray = arrayValue.asInstanceOf[Array[String]]
      var totalSize =
        Integer.BYTES + java.lang.Long.BYTES // gguf_metadata_value_type: uint32_t type; + Long.BYTES// uint64_t len;
      for (s <- stringArray) {
        totalSize += BuilderImpl.sizeOfStringValue(s)
      }
      totalSize
    } else {
      val array = arrayValue.asInstanceOf[Array[?]]
      java.lang.Integer.BYTES + java.lang.Long.BYTES + array.length * componentType.byteSize()
    }
    // Nested arrays are not supported yet.
//    assert(arrayValue.getClass.isArray && ((arrayValue.getClass.getComponentType eq classOf[String]) || arrayValue.getClass.getComponentType.isPrimitive))
//    Integer.BYTES // gguf_metadata_value_type: uint32_t component_type; + Long.BYTES// uint64_t len; + Array.getLength(arrayValue) * componentType.byteSize.toLong// gguf_metadata_value_t array[len];
  }

  private def computeTensorDataOffset = {
    var tensorDataOffset =
      Integer.BYTES // uint32_t MAGIC + Integer.BYTES// uint32_t version + Long.BYTES// uint64_t tensor_count + Long.BYTES// uint64_t metadata_kv_count;
//    var tensorDataOffset = java.lang.Integer.BYTES + java.lang.Integer.BYTES + java.lang.Long.BYTES + java.lang.Long.BYTES
    for (entry <- this.metadata) {
      val key = entry._1
      val value = entry._2
      tensorDataOffset += BuilderImpl.sizeOfStringValue(key)
      tensorDataOffset += sizeOfTaggedValue(key, value)
    }

    for (tensorInfo <- this.tensorInfos.values) {
      tensorDataOffset += BuilderImpl.sizeOfTensorInfo(tensorInfo)
    }
    val padding = GGUFImpl.padding(tensorDataOffset, getAlignment).toInt
    tensorDataOffset += padding
    tensorDataOffset
  }

  private[impl] def setTensors(newTensorInfos: mutable.LinkedHashMap[String, TensorInfo]) = {
    // Must preserve insertion order.
    assert(newTensorInfos.isInstanceOf[mutable.LinkedHashMap[?, ?]])
//    assert(newTensorInfos.allMatch((e: (String, TensorInfo)) => e.getKey == e.getValue.name))
    this.tensorInfos = newTensorInfos
    this
  }

  override def putTensor(tensorInfo: TensorInfo): BuilderImpl = {
    this.tensorInfos.put(tensorInfo.name, tensorInfo)
    this
  }

  override def removeTensor(tensorName: String): BuilderImpl = {
    this.tensorInfos.remove(tensorName)
    this
  }

  @SuppressWarnings(Array("unchecked"))
  override def getValue[T](targetClass: Class[T], key: String): T = {
    val value = this.metadata.get(key)
    if (value == null) {
      // value not found
      return null.asInstanceOf[T]
    }
    if (targetClass.isPrimitive) GGUFImpl.toBoxedClass(targetClass).cast(value).asInstanceOf[T]
    else targetClass.cast(value)
  }

  override def getTensor(tensorName: String): TensorInfo =
    this.tensorInfos.getOrElse(tensorName, null)

  override def getMetadataKeys = this.metadata.keySet.toSet

  override def getTensors = this.tensorInfos.values.toSeq

  override def getType(key: String): MetadataValueType = this.metadataTypes.getOrElse(key, null)

  override def getComponentType(key: String): MetadataValueType = {
    if (!this.metadata.contains(key)) return null
    this.metadataTypes.getOrElse(key + "[]", null)
  }

  override def removeKey(key: String): Builder = {
    this.metadata.remove(key)
    if (this.metadataTypes.remove(key) eq MetadataValueType.ARRAY)
      this.metadataTypes.remove(key + "[]")
    this
  }

  private def computeTensorOffsets = {
    var tensorOffset = 0L
    val newTensorInfos = new mutable.LinkedHashMap[String, TensorInfo]

    for (entry <- tensorInfos) {
      // Add padding, tensor start must be aligned.
      tensorOffset += GGUFImpl.padding(tensorOffset, getAlignment)
      val name = entry._1
      val tensorInfo = entry._2
      val ggmlType = tensorInfo.ggmlType
      val byteSize = ggmlType.byteSizeForShape(tensorInfo.shape*)
      newTensorInfos.put(name, TensorInfo.create(name, tensorInfo.shape, ggmlType, tensorOffset))
      tensorOffset += byteSize
    }
    newTensorInfos
  }

  override protected def putValue(
      key: String,
      valueType: MetadataValueType,
      value: Any
  ): BuilderImpl = {
//    Objects.requireNonNull(value)
    valueType match {

      case INT8 | UINT8 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Byte]))

      case INT16 | UINT16 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Short]))
      case INT32 | UINT32 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Int]))
      case FLOAT32 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Float]))
      case BOOL =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Boolean]))
      case STRING =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[String]))
      case INT64 | UINT64 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Long]))
      case FLOAT64 =>
        metadata.put(key, BuilderImpl.castTo(key, value, classOf[Double]))
      case ARRAY =>
        throw new IllegalArgumentException("use putKeyArrayOf instead")
    }
    metadataTypes.put(key, valueType)
    this
  }

  override protected def putArray(
      key: String,
      componentType: MetadataValueType,
      array: AnyRef
  ): BuilderImpl = {
//    Objects.requireNonNull(array)
    componentType match {
      case UINT8 => // fall-through
      case INT8 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Byte]]))
      case UINT16 => // fall-through
      case INT16 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Short]]))
      case UINT32 => // fall-through
      case INT32 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Int]]))
      case FLOAT32 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Float]]))
      case BOOL =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Boolean]]))
      case STRING =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[String]]))
      case UINT64 => // fall-through
      case INT64 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Long]]))
      case FLOAT64 =>
        metadata.put(key, BuilderImpl.castTo(key, array, classOf[Array[Double]]))
      case ARRAY =>
        throw new UnsupportedOperationException("array of arrays")
    }
    metadataTypes.put(key, MetadataValueType.ARRAY)
    metadataTypes.put(key + "[]", componentType)
    this
  }
}
