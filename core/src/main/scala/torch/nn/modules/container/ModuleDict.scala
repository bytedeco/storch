/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn
package modules
package container

import sourcecode.Name
import scala.annotation.varargs
import scala.collection.SortedSet
import scala.collection.immutable.TreeSeqMap
import scala.collection.mutable.{LinkedHashMap}
import scala.reflect.ClassTag.Any

/** Holds submodules in a list.
  *
  * It can be indexed like a regular Python list, but the modules it contains are properly
  * registered, and will be visible by all [[torch.nn.Module]] methods.
  *
  * @example
  *   ```scala
  *   class MyModule extends nn.Module:
  *     val linears = register( nn.ModuleList([nn.Linear(10, 10) for i in range(10)]) )
  *
  *     // ModuleList can act as an iterable, or be indexed using ints
  *     def forward(self, x) =
  *       var x_ = x.copy_(x)
  *       for l <- linears
  *         x_ = x_ + l(x_)
  *       x
  *   //  val tree = new TreeSeqMap[String, TensorModule[D]]()
  *   //  val modules = new Array[TensorModule[D]](nameModules.length)
  *   //  modules.zipWithIndex.foreach((module, index) =>
  *   //    this.register(module)(using Name(index.toString()))
  *   //  )
  *   //
  *   //  override def iterator: Iterator[TensorModule[D]] = modules.iterator
  *
  *   //    modules.foldLeft(input)((i, module) => module(i))
  *
  *   ```
  *
  * @see
  *   [[torch.nn.ModuleList https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html?highlight=modulelist#torch.nn.ModuleList]]
  * @see
  *   [[container ModuleList https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#ModuleList]]
  */
object ModuleDict {
  @varargs
  def apply[D <: DType](nameModules: (String, TensorModule[D])*): ModuleDict[D] = new ModuleDict(
    nameModules*
  )
}

@varargs
final class ModuleDict[D <: DType](@varargs var nameModules: (String, TensorModule[D])*)
    extends Module
    with TensorModule[D]
    with scala.collection.mutable.Iterable[(String, TensorModule[D])]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val moduleWithNameMap = new LinkedHashMap[String, TensorModule[D]]()
  nameModules.zipWithIndex.foreach((moduleNames, index) => {
    this.register(moduleNames._2)(using Name(moduleNames._1))
    moduleWithNameMap.put(moduleNames._1, moduleNames._2)
  })

  def iterator: Iterator[(String, TensorModule[D])] = moduleWithNameMap.iterator

  override def apply(input: Tensor[D]): Tensor[D] =
    nameModules.foldLeft(input)((tensor, nameModule) => nameModule._2(tensor))

  def forward(input: Tensor[D]): Tensor[D] = apply(input)

  override def toString = getClass().getSimpleName()

  /** Insert a given module before a given index in the list.
    *
    * @param index
    *   index to insert.
    * @param module
    *   module to insert
    * @return
    *   ModuleList[D] with new elements
    */

  def insert(preName: String, name: String, module: TensorModule[D]): ModuleDict[D] =

    val index = moduleWithNameMap.toIndexedSeq.zipWithIndex.find(_._1._1 == preName).get._2
    val (before, after) = moduleWithNameMap.splitAt(index)
    before.put(name, module)
    val all = before.toIndexedSeq ++ after.toIndexedSeq // (after.prepended(module))
    // TODO: not in Python code. Note other modules retain index, so we have repeats
    this.register(module)(using Name(name))
    // TODO: make modules list mutable?
    ModuleDict(all: _*)

  def append(name: String, module: TensorModule[D]): ModuleDict[D] =
    // TODO: not in Module
    // self.add_module(str(len(self)), module)
    // TODO: not in Python code
    val index = modules.length
    println(
      "ModuleDict append: module index: " + index
        .toString() + s"module name ${name}" + " modules map size: " + moduleWithNameMap.size
        .toString()
    )
    this.register(module)(using Name(name))
    moduleWithNameMap.put(name, module)
    val all = moduleWithNameMap.toIndexedSeq
    ModuleDict(all: _*)

  /** Appends modules from a Python iterable to the end of the list.
    *
    * @param modules
    *   iterable of modules to append
    * @return
    */
  def extend(newNameModulesSet: SortedSet[(String, TensorModule[D])]): ModuleDict[D] =

    newNameModulesSet.foreach((name, module) => {
      this.register(module)(using Name(name))
      moduleWithNameMap.put(name, module)

    })
    val all = nameModules ++ newNameModulesSet
    ModuleDict(all: _*)

  override def hasBias(): Boolean = moduleWithNameMap.values.exists(_.hasBias())

  def apply(name: String): torch.nn.modules.TensorModule[D] = moduleWithNameMap.get(name).get

  def length: Int = moduleWithNameMap.size

//    val all = modules.appended(module)
// TODO: make modules list mutable?

//  def extend(newModules: Iterable[TensorModule[D]]): ModuleList[D] =
//    // TODO: not in Module
//    // offset = len(self)
//    // for i, module in enumerate(modules):
//    //     self.add_module(str(offset + i), module)
//    // return self
//    // val offset = modules.length
//    val all = modules ++ newModules
//    // Not in Python
//    newModules.zipWithIndex.foreach((module, index) =>
//      this.register(module)(using Name(index.toString()))
//    )
//    // TODO: make modules list mutable?
//    ModuleList(all: _*)
// TODO: not in Module
// offset = len(self)
// for i, module in enumerate(modules):
//     self.add_module(str(offset + i), module)
// return self
// val offset = modules.length
//    val all = modules ++ newModules
// Not in Python
//    newModules.zipWithIndex.foreach((module, index) =>
//      this.register(module)(using Name(index.toString()))
//    )
// TODO: make modules list mutable?

//  def insert(index: Int, module: TensorModule[D]): ModuleList[D] =
//    val (before, after) = modules.splitAt(index)
//    val all = before ++ (after.prepended(module))
//    // TODO: not in Python code. Note other modules retain index, so we have repeats
//    this.register(module)(using Name(index.toString()))
//    // TODO: make modules list mutable?
//    ModuleList(all: _*)

//    val (before, after) = modules.splitAt(index)
/** Appends a given module to the end of the list.
  *
  * @param module
  *   module to append
  * @return
  *   ModuleList[D] with new elements
  */
//  def append(module: TensorModule[D]): ModuleList[D] =
//    // TODO: not in Module
//    // self.add_module(str(len(self)), module)
//    // TODO: not in Python code
//    val index = modules.length
//    this.register(module)(using Name(index.toString()))
//    val all = modules.appended(module)
//    // TODO: make modules list mutable?
//    ModuleList(all: _*)

//  def apply(i: Int): torch.nn.modules.TensorModule[D] = modules(i)

//  def apply(i: Int): torch.nn.modules.TensorModule[D] = modules(i)

//  nameModules.zipWithIndex.foreach( (moduleNames,index)=>{
//    this.register(moduleNames._2)(using Name(moduleNames._1))
//  })
//  override def iterator: Iterator[(String,TensorModule[D])] = nameModules.iterator
