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

import scala.annotation.varargs
import sourcecode.Name
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.SortedMap

/** A sequential container. Modules will be added to it in the order they are passed in the
  * constructor. Alternatively, you can also add modules with add_module. if add_module is used,
  * modules must be passed in the order they are to be added.
  * @param modules
  *   Any number of TensorModule[D] arguments.
  */
@varargs //override
final class Sequential[D <: FloatNN | ComplexNN: Default](
    val moduleSeq: SortedMap[String, TensorModule[D]] | TensorModule[D]*
) extends Module
    with TensorModule[D]:
  System.setProperty("org.bytedeco.javacpp.nopointergc", "true")
  val moduleList = ArrayBuffer[TensorModule[D]]()
  moduleSeq.zipWithIndex.foreach((module, index) =>
    module match
      case m: SortedMap[String, TensorModule[D]] =>
        m.foreach((name, moduleLayer) =>
          this.register(moduleLayer)(using Name(name.toString()))
          moduleList += moduleLayer
        )
      case m: TensorModule[D] =>
        this.register(m)(using Name(index.toString()))
        moduleList += m
  )

  def iterator: Iterator[TensorModule[D]] = moduleList.iterator

  def size(): Int = moduleList.length

  override def hasBias(): Boolean = moduleList.exists(_.hasBias())

  override def apply(input: Tensor[D]): Tensor[D] =
    moduleList.foldLeft(input)((i, module) => module(i))

  def forward(input: Tensor[D]): Tensor[D] = apply(input)

  def add_module(module: TensorModule[D]): Sequential[D] =

    val index = moduleList.length
    println(
      "Sequential add_module: module index: " + index
        .toString() + " modules length: " + moduleList.length.toString()
    )
    this.register(module)(using Name(index.toString()))
    moduleList += (module)
    this
    // TODO: make modules list mutable?
//    modules.append(module)
//    new Sequential(all: _*)

  override def toString = getClass().getSimpleName()

object Sequential {

//  @varargs
  def apply[D <: FloatNN | ComplexNN: Default](modules: TensorModule[D]*): Sequential[D] =
    new Sequential(modules*)

//  @varargs
  def apply[D <: FloatNN | ComplexNN: Default](
      modules: SortedMap[String, TensorModule[D]]
  ): Sequential[D] = new Sequential(modules)
}
