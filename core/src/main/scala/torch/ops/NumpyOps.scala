package torch
package ops

import org.bytedeco.javacpp.*
import spire.math.Complex
import torch.{DType, ScalaType, DTypeToScala, Tensor}
import torch.numpy.enums.DType as NumpyDType
import torch.numpy.matrix.NDArray

import scala.reflect.ClassTag
import torch.Layout.{Sparse, SparseBsc, SparseBsr, SparseCsc, SparseCsr, Strided}
import Device.CPU

private[torch] trait NumpyOps:

  def NDArrayToTensor[U <: ScalaType: ClassTag](
      data: NDArray[?],
      requires_grad: Boolean = false,
      device: Device = CPU
  ): Tensor[ScalaToDType[U]] = {
    require(data.getNdim <= 5, "Only 1D, 2D, and 3D, 4D, 5D arrays are supported")
    val shapeSize = data.getShape.size
    val tensor: Tensor[ScalaToDType[U]] = (data.getArray, shapeSize) match {
      case (singleDim: Array[U], 1) =>
        val dataSeq = singleDim.toSeq.asInstanceOf[Seq[U]]
        Tensor(dataSeq, Strided, device, requires_grad)
      case (twoDim: Array[Array[U]], 2) =>
        val dataSeq = twoDim.map((arr: Array[U]) => arr.toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (threeDim: Array[Array[Array[U]]], 3) =>
        val dataSeq = threeDim.map((arr: Array[Array[U]]) => arr.map(_.toSeq).toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (fourDim: Array[Array[Array[Array[U]]]], 4) =>
        val dataSeq =
          fourDim.map((arr: Array[Array[Array[U]]]) => arr.map(_.map(_.toSeq).toSeq).toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (fiveDim: Array[Array[Array[Array[Array[U]]]]], 5) =>
        val dataSeq = fiveDim
          .map((arr: Array[Array[Array[Array[U]]]]) =>
            arr.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq
          )
          .toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case _ => throw new IllegalArgumentException("Unsupported array dimension")
    }
    tensor
  }

  def apply[U <: ScalaType: ClassTag](
      NdArray: NDArray[U],
      requires_grad: Boolean = false,
      device: Device = CPU
  ): Tensor[ScalaToDType[U]] =
    Tensor.createFromNDArray(data = NdArray, requires_grad = requires_grad, device = device)

  def fromNDArray[U <: ScalaType: ClassTag](
      NdArray: NDArray[U],
      requires_grad: Boolean = false,
      device: Device = CPU
  ): Tensor[ScalaToDType[U]] =
    Tensor.createFromNDArray(data = NdArray, requires_grad = requires_grad, device = device)

  def fromNDArrayWithArray[U <: ScalaType: ClassTag](
      data: NDArray[U],
      requires_grad: Boolean = false,
      device: Device = CPU
  ): Tensor[ScalaToDType[U]] = {
    require(data.getNdim <= 5, "Only 1D, 2D, and 3D, 4D, 5D arrays are supported")
    val shapeSize = data.getShape.size
    val tensor: Tensor[ScalaToDType[U]] = (data.getArray, shapeSize) match {
      case (singleDim: Array[U], 1) =>
        val dataSeq = singleDim.toSeq.asInstanceOf[Seq[U]]
        Tensor(dataSeq, Strided, device, requires_grad)
      case (twoDim: Array[Array[U]], 2) =>
        val dataSeq = twoDim.map((arr: Array[U]) => arr.toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (threeDim: Array[Array[Array[U]]], 3) =>
        val dataSeq = threeDim.map((arr: Array[Array[U]]) => arr.map(_.toSeq).toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (fourDim: Array[Array[Array[Array[U]]]], 4) =>
        val dataSeq =
          fourDim.map((arr: Array[Array[Array[U]]]) => arr.map(_.map(_.toSeq).toSeq).toSeq).toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case (fiveDim: Array[Array[Array[Array[Array[U]]]]], 5) =>
        val dataSeq = fiveDim
          .map((arr: Array[Array[Array[Array[U]]]]) =>
            arr.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq
          )
          .toSeq
        Tensor(dataSeq, Strided, device, requires_grad)
      case _ => throw new IllegalArgumentException("Unsupported array dimension")
    }
    tensor
  }

  /** *
    *
    * @param arr
    * @tparam U
    * @return
    */
  def arrayToSeq[U <: ScalaType: ClassTag](
      arr: Array[U] | Array[Array[U]] | Array[Array[Array[U]]]
  ): U | Seq[U] | Seq[Seq[U]] | Seq[Seq[Seq[U]]] = {
    arr match {
      case singleDim: Array[U]              => singleDim.toSeq
      case twoDim: Array[Array[U]]          => twoDim.map(_.toSeq).toSeq
      case threeDim: Array[Array[Array[U]]] => threeDim.map(_.map(_.toSeq).toSeq).toSeq
      case _ => throw new IllegalArgumentException("Unsupported array dimension")
    }
  }

  /** * Convert a torch.Tensor to a numpy.NDArray
    * @param tensor
    * @tparam D
    * @return
    */
  def toNDArray[D <: DType](
      tensor: Tensor[D]
  )(using ct: ClassTag[DTypeToScala[D]]): NDArray[DTypeToScala[D]] = {
    require(tensor.ndimension().toInt <= 5, "Only 1D, 2D, and 3D, 4D, 5D tensors are supported")
    val shape = tensor.sizes().map(_.toLong).toArray
    var npDtype: NumpyDType = null
    val data = tensor.dtype match {
      case DType.uint8 =>
        npDtype = NumpyDType.UInt8
        val byteData = new Array[Byte](tensor.native.numel().toInt)
        val tensorPointer = new BytePointer(tensor.native.data_ptr_byte())
        tensorPointer.get(byteData)
        byteData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.int8 =>
        npDtype = NumpyDType.Int8
        val byteData = new Array[Byte](tensor.native.numel().toInt)
        val tensorPointer = new BytePointer(tensor.native.data_ptr_byte())
        tensorPointer.get(byteData)
        byteData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.int16 =>
        npDtype = NumpyDType.Int16
        val shortData = new Array[Short](tensor.native.numel().toInt)
        val tensorPointer = new ShortPointer(tensor.native.data_ptr_short())
        tensorPointer.get(shortData)
        shortData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.int32 =>
        npDtype = NumpyDType.Int32
        val intData = new Array[Int](tensor.native.numel().toInt)
        val tensorPointer = new IntPointer(tensor.native.data_ptr_int())
        tensorPointer.get(intData)
        intData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.int64 =>
        npDtype = NumpyDType.Int64
        val longData = new Array[Long](tensor.native.numel().toInt)
        val tensorPointer = new LongPointer(tensor.native.data_ptr_long())
        tensorPointer.get(longData)
        longData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.float16 =>
        npDtype = NumpyDType.Float16
        val floatData = new Array[Float](tensor.native.numel().toInt)
        val tensorPointer = new FloatPointer(tensor.native.data_ptr_float())
        tensorPointer.get(floatData)
        floatData.asInstanceOf[Array[DTypeToScala[D]]]
        throw new UnsupportedOperationException("Float16 is not supported yet")
      case DType.float32 =>
        npDtype = NumpyDType.Float32
        val floatData = new Array[Float](tensor.native.numel().toInt)
        val tensorPointer = new FloatPointer(tensor.native.data_ptr_float())
        tensorPointer.get(floatData)
        floatData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.float64 =>
        npDtype = NumpyDType.Float64
        val doubleData = new Array[Double](tensor.native.numel().toInt)
        val tensorPointer = new DoublePointer(tensor.native.data_ptr_double())
        tensorPointer.get(doubleData)
        doubleData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.complex32 =>
        npDtype = NumpyDType.UInt32
        val floatData = new Array[Float](tensor.native.numel().toInt * 2)
        val tensorPointer = new FloatPointer(tensor.native.data_ptr_float())
        tensorPointer.get(floatData)
        val complexData = floatData.grouped(2).map { case Array(r, i) => Complex(r, i) }.toArray
        complexData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.complex64 =>
        npDtype = NumpyDType.UInt64
        val doubleData = new Array[Double](tensor.native.numel().toInt * 2)
        val tensorPointer = new DoublePointer(tensor.native.data_ptr_double())
        tensorPointer.get(doubleData)
        val complexData = doubleData.grouped(2).map { case Array(r, i) => Complex(r, i) }.toArray
        complexData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.complex128 =>
        throw new UnsupportedOperationException("Complex128 is not supported yet")
      case DType.bool =>
        npDtype = NumpyDType.Bool
        val boolData = new Array[Boolean](tensor.native.numel().toInt)
        val boolPtr = new BoolPointer(tensor.native.data_ptr_bool())
        for (i <- 0 until boolData.length) {
          boolData(i) = boolPtr.get(i)
        }
        boolData.asInstanceOf[Array[DTypeToScala[D]]]
      case DType.qint8 =>
        throw new UnsupportedOperationException("QInt8 is not supported yet")
      case DType.quint8 =>
        throw new UnsupportedOperationException("QUInt8 is not supported yet")
      case DType.qint32 =>
        throw new UnsupportedOperationException("QInt32 is not supported yet")
      case DType.bfloat16 =>
        throw new UnsupportedOperationException("BFloat16 is not supported yet")
      case DType.quint4x2 =>
        throw new UnsupportedOperationException("QUInt4x2 is not supported yet")
      case DType.quint2x4 =>
        throw new UnsupportedOperationException("QUInt2x4 is not supported yet")
      case DType.bits1x8 =>
        throw new UnsupportedOperationException("Bits1x8 is not supported yet")
      case DType.bits2x4 =>
        throw new UnsupportedOperationException("Bits2x4 is not supported yet")
      case DType.bits4x2 =>
        throw new UnsupportedOperationException("Bits4x2 is not supported yet")
      case DType.bits8 =>
        throw new UnsupportedOperationException("Bits8 is not supported yet")
      case DType.bits16 =>
        throw new UnsupportedOperationException("Bits16 is not supported yet")
      case DType.float8_e5m2 =>
        throw new UnsupportedOperationException("Float8_e5m2 is not supported yet")
      case DType.float8_e4m3fn =>
        throw new UnsupportedOperationException("Float8_e4m3fn is not supported yet")
      case DType.undefined =>
        throw new UnsupportedOperationException("Undefined is not supported")
      case DType.numoptions =>
        throw new UnsupportedOperationException("NumOptions is not supported")
    }

    new NDArray[DTypeToScala[D]](
      data = data,
      shape = shape.map(_.toInt),
      ndim = shape.size,
      dType = npDtype
    ).reshape(shape.map(_.toInt)*)

  }

//val tensor2: Tensor[ScalaToDType[U]] = data.getArray match {
//  case singleDim: Array[U] =>
//    val dataSeq = singleDim.toSeq
//    Tensor(dataSeq, Strided, device, requires_grad)
//  case twoDim: Array[Array[U]] =>
//    val dataSeq = twoDim.map((arr: Array[U]) => arr.toSeq).toSeq
//    Tensor(dataSeq, Strided, device, requires_grad)
//  case threeDim: Array[Array[Array[U]]] =>
//    val dataSeq = threeDim.map((arr: Array[Array[U]]) => arr.map(_.toSeq).toSeq).toSeq
//    Tensor(dataSeq, Strided, device, requires_grad)
//  case fourDim: Array[Array[Array[Array[U]]]] =>
//    val dataSeq = fourDim.map((arr: Array[Array[Array[U]]]) => arr.map(_.map(_.toSeq).toSeq).toSeq).toSeq
//    Tensor(dataSeq, Strided, device, requires_grad)
//  case fiveDim: Array[Array[Array[Array[Array[U]]]]] =>
//    val dataSeq = fiveDim.map((arr: Array[Array[Array[Array[U]]]]) => arr.map(_.map(_.map(_.toSeq).toSeq).toSeq).toSeq).toSeq
//    Tensor(dataSeq, Strided, device, requires_grad)
//  case _ => throw new IllegalArgumentException("Unsupported array dimension")
//}
