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
package utils
package data

import torch.DType

/** Wraps a pair of tensors as a Seq.
  *
  * Each sample will be retrieved by indexing tensors along the first dimension.
  */
// TODO can we generalize this to tuples of arbitrary size?
trait NormalTensorDataset[Input <: DType, Target <: DType]
    extends IndexedSeq[(Tensor[Input], Tensor[Target])] {
  def features: Tensor[Input]

  def targets: Tensor[Target]

//  def length: Long

  def apply(idx: Int): (Tensor[Input], Tensor[Target])

  def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = apply(idx)

  def get_item(idx: Int): (Tensor[Input], Tensor[Target]) = getItem(idx)
}

object NormalTensorDataset {
  def apply[Input <: DType, Target <: DType](
      _features: Tensor[Input],
      _targets: Tensor[Target]
  ): NormalTensorDataset[Input, Target] = new NormalTensorDataset[Input, Target] {
    val features = _features
    val targets = _targets

    require(features.size.length > 0)
    require(features.size.head == targets.size.head)

    override def apply(i: Int): (Tensor[Input], Tensor[Target]) = (features(i), targets(i))

    override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = (features(idx), targets(idx))

    override def length: Int = features.size.head

    override def toString(): String =
      s"TensorDataset(features=${features.info}, targets=${targets.info})"
  }
}
