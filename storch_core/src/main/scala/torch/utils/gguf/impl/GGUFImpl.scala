package torch.utils.gguf.impl

import torch.utils.gguf.MetadataValueType.*
import torch.utils.gguf.{GGUF, MetadataValueType, TensorInfo}

import scala.collection.mutable

object GGUFImpl {
  private[impl] def toBoxedClass[T](primitiveClass: Class[T]): Class[?] = {
    primitiveClass match
      case cls if cls == classOf[Boolean] => classOf[java.lang.Boolean]
      case cls if cls == classOf[Byte]    => classOf[java.lang.Byte]
      case cls if cls == classOf[Char]    => classOf[java.lang.Character]
      case cls if cls == classOf[Short]   => classOf[java.lang.Short]
      case cls if cls == classOf[Int]     => classOf[java.lang.Integer]
      case cls if cls == classOf[Long]    => classOf[java.lang.Long]
      case cls if cls == classOf[Float]   => classOf[java.lang.Float]
      case cls if cls == classOf[Double]  => classOf[java.lang.Double]
      case cls if cls == classOf[Unit]    => classOf[java.lang.Void]
      case _ => throw new IllegalArgumentException("not a primitive class " + primitiveClass)

//    if (primitiveClass eq classOf[Boolean]) return classOf[Boolean]
//    if (primitiveClass eq classOf[Byte]) return classOf[Byte]
//    if (primitiveClass eq classOf[Char]) return classOf[Character]
//    if (primitiveClass eq classOf[Short]) return classOf[Short]
//    if (primitiveClass eq classOf[Int]) return classOf[Integer]
//    if (primitiveClass eq classOf[Long]) return classOf[Long]
//    if (primitiveClass eq classOf[Float]) return classOf[Float]
//    if (primitiveClass eq classOf[Double]) return classOf[Double]
//    if (primitiveClass eq classOf[Unit]) return classOf[Void]
//    throw new IllegalArgumentException("not a primitive class " + primitiveClass)
  }

  private[impl] def padding(position: Long, alignment: Long) = {
    val nextAlignedPosition = (position + alignment - 1) / alignment * alignment
    nextAlignedPosition - position
  }
}

final class GGUFImpl(
    val version: Int,
    val tensorDataOffset: Long,
    metadata: mutable.Map[String, Any],
    metadataTypes: mutable.Map[String, MetadataValueType],
    tensorInfos: mutable.Map[String, TensorInfo]
) extends GGUF {
//  this.metadata = Collections.unmodifiableMap(new mutable.LinkedHashMap[String, AnyRef](metadata))
//  this.metadataTypes = Collections.unmodifiableMap(new mutable.LinkedHashMap[String, MetadataValueType](metadataTypes))
//  this.tensorInfos = Collections.unmodifiableMap(new mutable.LinkedHashMap[String, TensorInfo](tensorInfos))
//  final private var metadata: mutable.Map[String, AnyRef] = new mutable.LinkedHashMap[String, AnyRef](metadata)
//  final private var metadataTypes: mutable.Map[String, MetadataValueType] = new mutable.LinkedHashMap[String, MetadataValueType](metadataTypes)
//  final private var tensorInfos: mutable.Map[String, TensorInfo] = new mutable.LinkedHashMap[String, TensorInfo](tensorInfos)

  override def getVersion: Int = this.version

  override def getTensorDataOffset: Long = this.tensorDataOffset

  override def getMetadataKeys = this.metadata.keySet.toSet

  @SuppressWarnings(Array("unchecked"))
  override def getValue[T](targetClass: Class[T], key: String): T = {
    val value = this.metadata.getOrElse(key, null)
    if (value == null) {
      // value not found
      return null.asInstanceOf[T]
    }
    if (targetClass.isPrimitive) GGUFImpl.toBoxedClass(targetClass).cast(value).asInstanceOf[T]
    else targetClass.cast(value)
  }

  override def getType(key: String): MetadataValueType = this.metadataTypes.getOrElse(key, null)

  override def getComponentType(key: String): MetadataValueType = {
    if (!this.metadata.contains(key)) return null
    this.metadataTypes.getOrElse(key + "[]", null)
  }

  override def getTensor(tensorName: String): TensorInfo =
    this.tensorInfos.getOrElse(tensorName, null)

  override def getTensors: Seq[TensorInfo] = this.tensorInfos.values.toSeq
}
