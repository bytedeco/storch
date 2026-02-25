package torch.utils.gguf

import java.util
import java.util.Objects

/** Represents metadata about a tensor stored in a GGUF file. <p> A tensor is a multidimensional
  * array of values used in machine learning models. This class stores key information about a
  * tensor including its name, shape, data type, and location/offset with respect to {@link GGUF#
  * getTensorDataOffset ( )}.
  */
//trait TensorInfo:
//  def name(): String
//  def shape(): Array[Long]
//  def ggmlType(): GGMLType
//  def offset(): Long
//  def create(name: String, shape: Array[Long], ggmlType: GGMLType, offset: Long): TensorInfo

object TensorInfo {

  /** Constructs a new {@link TensorInfo} with the specified parameters.
    *
    * @param name
    *   the name identifier of the tensor
    * @param shape
    *   the dimensions of the tensor as an array of longs
    * @param ggmlType
    *   the data type of the tensor elements
    * @param offset
    *   the byte offset where this tensor's data begins in the file
    */
  def create(name: String, shape: Array[Long], ggmlType: GGMLType, offset: Long) =
    new TensorInfo(name, shape, ggmlType, offset)
}

final class TensorInfo(
    /** The name identifier of the tensor.
      */
    val name: String,

    /** The dimensions of the tensor. For example, [768, 32000] represents a 2D tensor with 768 rows
      * and 32000 columns.
      */
    val shape: Array[Long],

    /** The data type of the tensor elements e.g. {@link GGMLType# F32}, {@link GGMLType# Q4_0}.
      */
    val ggmlType: GGMLType,

    /** The byte offset where this tensor's data begins with respect to {@link GGUF#
      * getTensorDataOffset ( )} in the GGUF file.
      */
    val offset: Long
) {

  /** Returns the name identifier of the tensor.
    *
    * @return
    *   the tensor name
    */
  def getName: String = name

  /** Returns the dimensions of the tensor. <p> The returned array represents the size of each
    * dimension. For example, [768, 32000] represents a 2D tensor with 768 rows and 32000 columns.
    *
    * @return
    *   the tensor's shape as an array of longs
    */
  def getShape: Array[Long] = shape

  /** Returns the data type of the tensor.
    *
    * <p> Tensors can be {@link GGMLType# isQuantized ( ) quantized} e.g. {@link GGMLType# Q8_0}.
    *
    * @return
    *   the GGML data type
    */
  def getGgmlType: GGMLType = ggmlType

  /** Returns the byte offset where this tensor's data begins with respect to {@link GGUF#
    * getTensorDataOffset ( )}.
    *
    * @return
    *   the byte offset of the tensor data
    */
  def getOffset: Long = offset

  /** Compares this {@link TensorInfo} with another object for equality. <p> Two TensorInfo objects
    * are considered equal if they have the same name, shape, type, and offset.
    *
    * @param other
    *   the object to compare with
    * @return
    *   true if the objects are equal, false otherwise
    */
  override def equals(other: Any): Boolean = {
    if (this eq other.asInstanceOf[TensorInfo]) return true
    if (other.isInstanceOf[TensorInfo]) {
      val that = other.asInstanceOf[TensorInfo]
      offset == that.offset && Objects.equals(name, that.name) && Objects.deepEquals(
        shape,
        that.shape
      ) && (ggmlType eq that.ggmlType)
    } else false
  }

  /** Returns a hash code value for this {@link TensorInfo}.
    *
    * @return
    *   a hash code value for this object
    */
  override def hashCode: Int = Objects.hash(name, util.Arrays.hashCode(shape), ggmlType, offset)

  /** Returns a string representation of this {@link TensorInfo}. <p> The string includes the
    * tensor's name, shape, type, and offset in hexadecimal. For example:
    * <i>"TensorInfo{name='token_embd.weight', shape=[768, 32000], ggmlType=F32,
    * offset=0x12300}"</i>
    *
    * @return
    *   a string representation of this object
    */
  override def toString: String =
    "TensorInfo{" + "name='" + name + '\'' + ", shape=" + util.Arrays.toString(
      shape
    ) + ", ggmlType=" + ggmlType + ", offset=" + "0x" + java.lang.Long.toHexString(offset) + '}'
}
