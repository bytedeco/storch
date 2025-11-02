package torch
package nn
package modules

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  CompilationUnit,
  ExtraFilesMap,
  IValue,
  Type,
  IValueVector,
  InputArchive,
  JitModule,
  OutputArchive,
  QualifiedName,
  StringIValueMap,
  named_buffer_iterator,
  named_attribute_iterator,
  named_parameter_iterator
}
import torch.internal.NativeConverters.fromNative
import org.bytedeco.javacpp.{BytePointer, Pointer}
import org.bytedeco.pytorch.global.torch as torchNative
import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}
import scala.collection.mutable.{ListBuffer, HashMap as MutaHashMap}
import java.io.{InputStream, OutputStream, ByteArrayOutputStream}
import java.nio.ByteBuffer
object JITModule {

//  def load(fileName: String): JitModule = {
//    val jitModule = new JitModule(new QualifiedName(fileName))
//    jitModule.load(fileName)
//    jitModule
//  }

  def load(fileName: String): JitModule = torchNative.load(fileName)

  def apply(
      qualifiedName: String,
      compilationUnit: CompilationUnit,
      shouldMangle: Boolean = false
  ): JITModule = {
    val nativeJitModule =
      new JitModule(new QualifiedName(qualifiedName), compilationUnit, shouldMangle)
    new JITModule(nativeJitModule)
  }
}
class JITModule(nativeModule: JitModule) {

  protected[torch] var nativeJitModule: JitModule = nativeModule

//  def this(qualifiedName: String,
//           compilationUnit: CompilationUnit,
//           shouldMangle: Boolean = false) = {
//    this.nativeJitModule =  new JitModule(new QualifiedName(qualifiedName), compilationUnit, shouldMangle)
//  }

//  def this(nativeModule: JitModule) ={
//    this.nativeJitModule = nativeModule
//  }

  def forward(input: IValueVector): IValue = nativeJitModule.forward(input)

  def forward(input: IValueVector, map: Map[String, IValue]): IValue = {
    val kwargs: StringIValueMap = new StringIValueMap()
    for ((k, v) <- map.toList) {
      kwargs.put(new BytePointer(k), v)
    }
//    val pred = fromNative(nativeJitModule.forward(input, kwargs).toTensor)
    nativeJitModule.forward(input, kwargs)
  }

  private[torch] def nativeModules(): JitModule = nativeJitModule

  private var childModules: TreeSeqMap[String, JitModule] = TreeSeqMap.empty

//  private var nativeJitModule: JitModule =
//    new JitModule(new QualifiedName(qualifiedName), compilationUnit, shouldMangle)
//
  def apply(predictTensor: Tensor[?]): Tensor[?] = {
    val vector = new IValueVector(predictTensor.native)
    val pred = fromNative(nativeJitModule.forward(vector).toTensor)
    pred
  }

  def apply(predictSeq: Seq[Int]): Tensor[?] = {

    val vector = new IValueVector(torch.ones(predictSeq).native)
    val pred = fromNative(nativeJitModule.forward(vector).toTensor())
    pred
  }

  def apply(predictTensor: Tensor[?], parameter: MutaHashMap[String, Tensor[?]]): Tensor[?] = {

    val vector = new IValueVector(predictTensor.native)
    val kwargs: StringIValueMap = new StringIValueMap()
    for ((k, v) <- parameter.toList) {
      kwargs.put(new BytePointer(k), new IValue(v.native))
    }
    val pred = fromNative(nativeJitModule.forward(vector, kwargs).toTensor)
    pred
  }

  def register_buffer(name: String, tensor: Tensor[?]): Unit =
    nativeJitModule.register_buffer(name, tensor.native)

  def register_parameter(name: String, tensor: Tensor[?], is_buffer: Boolean): Unit = {
    nativeJitModule.register_parameter(name, tensor.native, is_buffer)
  }

  def register_attribute(name: String, typez: Type.TypePtr, value: IValue): Unit = {
    nativeJitModule.register_attribute(name, typez, value)

  }

  def register_attribute(
      name: String,
      typez: Type.TypePtr,
      value: IValue,
      is_buffer: Boolean,
      persistent: Boolean
  ): Unit = {
    nativeJitModule.register_attribute(name, typez, value, is_buffer, persistent)

  }
  def register_module(name: String, module: JitModule) =
    nativeJitModule.register_module(name, module)
  def save(fileName: String) = nativeJitModule.save(fileName)

  def save(fileName: String, extraFiles: ExtraFilesMap) = nativeJitModule.save(fileName, extraFiles)
//https://pytorch.org/docs/stable/generated/torch.jit.save.html
  def save(outputStream: ByteArrayOutputStream) = {
    val byteArray: Array[Byte] = outputStream.toByteArray()
    val bytePointer: BytePointer = new BytePointer(byteArray*)
    nativeJitModule.save(bytePointer)
  }

  def save(outputStream: ByteArrayOutputStream, extraFiles: ExtraFilesMap) = {

    val byteArray: Array[Byte] = outputStream.toByteArray()
    val bytePointer: BytePointer = new BytePointer(byteArray*)
    nativeJitModule.save(bytePointer, extraFiles)
  }

  def save_for_mobile(fileName: String) = nativeJitModule._save_for_mobile(fileName)

  def deepcopy() = nativeJitModule.deepcopy()

  def clone_native() = nativeJitModule.clone()

  def store_traced_inputs(funcName: String, inputs: Tensor[?]) = {
    val vector = new IValueVector(inputs.native)
    nativeJitModule.store_traced_inputs(funcName, vector)
  }

  def set_delete_memory(delete_mem: ByteBuffer) = nativeJitModule.set_delete_memory(delete_mem)

  def retrieve_traced_inputs = nativeJitModule.retrieve_traced_inputs()

  def eval() = nativeJitModule.eval()

  def train() = nativeJitModule.train()

  def train(on: Boolean) = nativeJitModule.train(on)

  def is_training = nativeJitModule.is_training()

  def to(device: Device): this.type =
    nativeJitModule.to(device.toNative, false)
    this

  def named_buffers(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val buffers = nativeJitModule.named_buffers(recurse)
    var ele = buffers.begin()
    val buff = new ListBuffer[named_buffer_iterator]()
    while (ele != buffers.end()) {
      buff.append(ele)
      ele.increment()
    }
    TreeSeqMap.from((0 until buffers.size().toInt).map { i =>
      val item = buff(i).access()
      (item.name().getString(), fromNative[DType](item.value()))
    })

  def named_parameters(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val params = nativeJitModule.named_parameters(recurse)
    var ele = params.begin()
    val buffer = new ListBuffer[named_parameter_iterator]()
    while (ele != params.end()) {
      buffer.append(ele)
      ele.increment()
    }
    TreeSeqMap.from((0 until params.size().toInt).map { i =>
      val item = buffer(i).access()
      (item.name().getString(), fromNative[DType](item.value()))
    })

  def load_state_dict(stateDict: Map[String, Tensor[DType]]): Unit =
    val tensorsToLoad = named_parameters() ++ named_buffers()
    // assert(stateDict.keySet -- tensorsToLoad.keySet == Set.empty, s"keys missing in state dict: ${tensorsToLoad.keySet -- stateDict.keySet}")
    for ((key, param) <- tensorsToLoad if stateDict.contains(key))
      noGrad {
        param.copy_(stateDict(key))
      }

  def set_optimized(optimized: Boolean) = nativeJitModule.set_optimized(optimized)

  def is_optimized(): Boolean = nativeJitModule.is_optimized()

  def dump(pmb: Boolean, pav: Boolean, ppv: Boolean) = nativeJitModule.dump(pmb, pav, ppv)

  def attributes(recurse: Boolean = true) = {

    val attr = nativeJitModule.attributes(recurse)
    var ele = attr.begin()
    val buffer = new ListBuffer[IValue]()
    while (ele != attr.end()) {
      buffer.append(ele.access())
      ele.increment()
    }
    ArraySeq.unsafeWrapArray(buffer.toArray)
  }

  def named_attributes(recurse: Boolean = true) = {

    val nameAttr = nativeJitModule.named_attributes(recurse)
    var ele = nameAttr.begin()
    val buffer = new ListBuffer[named_attribute_iterator]()
    while (ele != nameAttr.end()) {
      buffer.append(ele)
      ele.increment()
    }
    TreeSeqMap.from((0 until nameAttr.size().toInt).map { i =>
      val item = buffer(i).access()
      (item.name().getString(), item.value())
    })
  }

  def namedParameters(recurse: Boolean = true) = nativeJitModule.named_parameters(recurse)

  def parameters: Seq[Tensor[?]] = parameters(recurse = true)

  def parameters(recurse: Boolean): Seq[Tensor[?]] = {
    val params = nativeJitModule.parameters(recurse)
    var ele = params.begin()
    val buffer = new ListBuffer[Tensor[?]]()
    while (ele != params.end()) {
      buffer.append(fromNative[DType](ele.access()))
      ele.increment()
    }
    ArraySeq.unsafeWrapArray(buffer.toArray) // .map(fromNative[DType])
  }

  def buffers(recurse: Boolean = true) = nativeJitModule.buffers(recurse)

  def namedBuffers(recurse: Boolean = true) = nativeJitModule.named_buffers(recurse)

  def children() = nativeJitModule.children()

  def named_children() = nativeJitModule.named_children()

  def modules() = nativeJitModule.modules()

  def named_modules() = nativeJitModule.named_modules()

}

//      // Create a vector of inputs.
//        IValueVector inputs = new IValueVector();
//        inputs.push_back(new IValue(ones(1, 3, 224, 224)));
//        // Execute the model and turn its output into a tensor.
//        Tensor output = module.forward(inputs).toTensor();
//        print(output.slice(/*dim=*/1, /*start=*/new LongOptional(0), /*end=*/new LongOptional(5), /*step=*/1));

//  def parameters(recurse: Boolean =true) = nativeJitModule.parameters(recurse)
