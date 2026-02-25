//package torch.nn.functional
//
//import org.bytedeco.javacpp.*
//import org.bytedeco.javacpp.annotation.{Cast, Name, Namespace, Platform}
//
//import scala.reflect.ClassTag
//
//abstract class ArrayRef[T: ClassTag, P <: Pointer](list_data: Array[T], final val dataType: DataType[T]) extends Pointer(null.asInstanceOf[Pointer]) {
//  val size: Int = list_data.length
//  def getData(): P
//  def toArray: Array[T] = {
//    getData() match {
//      case d: FloatPointer => Array.range(0, size).map(d.get(_).asInstanceOf[T]).toArray
//      case d: IntPointer => Array.range(0, size).map(d.get(_).asInstanceOf[T]).toArray
//      case d: LongPointer => Array.range(0, size).map(d.get(_).asInstanceOf[T]).toArray
//      case d: DoublePointer => Array.range(0, size).map(d.get(_).asInstanceOf[T]).toArray
//      case d: BytePointer => Array.range(0, size).map(d.get(_).asInstanceOf[T]).toArray
//    }
//  }
//}
//
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<float>")) class ArrayRefFloat(list_data: Array[Float]) extends  ArrayRef[Float, FloatPointer](list_data,FLOAT32) {
//
//  @native def allocate(@Cast(Array("float*")) d: FloatPointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new FloatPointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("float*")) def data(): FloatPointer
//
//  override def getData(): FloatPointer = data()
//
//}
//
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<int>")) class ArrayRefInt(list_data: Array[Int]) extends  ArrayRef[Int, IntPointer](list_data,INT32) {
//
//  @native def allocate(@Cast(Array("int*")) d: IntPointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new IntPointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("int*")) def data(): IntPointer
//
//  override def getData(): IntPointer = data()
//
//}
//
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<long>")) class ArrayRefLong(list_data: Array[Long]) extends  ArrayRef[Long, LongPointer](list_data,INT64) {
//
//
//  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new LongPointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("long long int*")) def data(): LongPointer
//
//  override def getData(): LongPointer = data()
//
//}
//
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<double>")) class ArrayRefDouble(list_data: Array[Double]) extends  ArrayRef[Double, DoublePointer](list_data,FLOAT64) {
//
//
//  @native def allocate(@Cast(Array("double*")) d: DoublePointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new DoublePointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("double*")) def data(): DoublePointer
//
//  override def getData(): DoublePointer = data()
//
//}
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<uint8_t>")) class ArrayRefByte(list_data: Array[Byte]) extends  ArrayRef[Byte, BytePointer](list_data,INT8) {
//
//  @native def allocate(@Cast(Array("uint8_t*")) d: BytePointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new BytePointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("int8_t*")) def data(): BytePointer
//
//  override def getData(): BytePointer = data()
//
//}
//
//
//
//
//
//object ArrayRef {
//  def apply[T](data: Array[T]): ArrayRef[T, _] = data.head match {
//    case v: Int => new ArrayRefInt(data.asInstanceOf[Array[Int]]).asInstanceOf[ArrayRef[T, _]]
//    case v: Float => new ArrayRefFloat(data.asInstanceOf[Array[Float]]).asInstanceOf[ArrayRef[T, _]]
//    case v: Long => new ArrayRefLong(data.asInstanceOf[Array[Long]]).asInstanceOf[ArrayRef[T, _]]
//    case v: Double => new ArrayRefDouble(data.asInstanceOf[Array[Double]]).asInstanceOf[ArrayRef[T, _]]
//    case v: Byte => new ArrayRefByte(data.asInstanceOf[Array[Byte]]).asInstanceOf[ArrayRef[T, _]]
//  }
//}
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<long>")) class IntList(list_data: Array[Long]) extends Pointer(null.asInstanceOf[Pointer]) {
//  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new LongPointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("long*")) def data(): LongPointer
//}
//
//
//@Platform(include = Array("c10/util/ArrayRef.h"))
//@Namespace("c10") @Name(Array("ArrayRef<long>")) class IntArrayRef(list_data: Array[Long]) extends Pointer(null.asInstanceOf[Pointer]) {
//  @native def allocate(@Cast(Array("long*")) d: LongPointer, @Cast(Array("size_t")) length: Int): Unit
//  allocate(new LongPointer(list_data:_*), list_data.length)
//
//  @native @Cast(Array("long*")) def data(): LongPointer
//}
//
//
//object IntList {
//  def apply(list_data: Array[Long]): IntList = new IntList(list_data)
//  def apply(list_data: Array[Int]): IntList = new IntList(list_data.map(_.toLong))
//}
//
//@Platform(include = Array("torch/all.h"))
//@Namespace("at") @Name(Array("ArrayRef<at::Tensor>")) class TensorList[T, TT <: TensorType](list_data: Array[Tensor[T, TT]]) extends PointerPointer[Tensor[T, TT]] {
//  @native def allocate(@Cast(Array("at::Tensor*")) d: PointerPointer[Tensor[T, TT]], @Cast(Array("size_t")) length: Int): Unit
//  allocate(new PointerPointer[Tensor[T, TT]](list_data:_*), list_data.length)
//
//  @native @Cast(Array("at::Tensor*")) def data(): PointerPointer[Tensor[T, TT]]
//}
