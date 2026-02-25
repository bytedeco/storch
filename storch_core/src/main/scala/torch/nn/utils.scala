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

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.TensorVector
import org.bytedeco.pytorch.TensorListIterator
import torch.internal.NativeConverters.fromNative
import scala.annotation.StaticAnnotation

object utils:
  def clipGradNorm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double,
      norm_type: Double = 2.0,
      error_if_nonfinite: Boolean = false
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm,
      norm_type,
      error_if_nonfinite
    )

  def clip_grad_norm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm
    )

  def clip_grad_norm_(
      parameters: Tensor[?],
      max_norm: Double
  ): Double =
    torchNative.clip_grad_norm_(
      parameters.native,
      max_norm
    )

  def clip_grad_norm_(
      parameters: Tensor[?],
      max_norm: Double,
      norm_type: Double = 2.0,
      error_if_nonfinite: Boolean = false
  ): Double =
    torchNative.clip_grad_norm_(
      parameters.native,
      max_norm,
      norm_type,
      error_if_nonfinite
    )
  def clip_grad_norm_(
      parameters: Seq[Tensor[?]],
      max_norm: Double,
      norm_type: Double,
      error_if_nonfinite: Boolean
  ): Double =
    torchNative.clip_grad_norm_(
      TensorVector(parameters.map(_.native).toArray*),
      max_norm,
      norm_type,
      error_if_nonfinite
    )

  def clip_grad_value_(
      parameters: Seq[Tensor[?]],
      clip_value: Double
  ): Unit =
    torchNative.clip_grad_value_(
      TensorVector(parameters.map(_.native).toArray*),
      clip_value
    )

  def clip_grad_value_(
      parameters: Tensor[?],
      clip_value: Double
  ): Unit =
    torchNative.clip_grad_value_(
      parameters.native,
      clip_value
    )

  def gammainc[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      other: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gammainc(input.native, other.native))

  def gammaincc[D1 <: DType, D2 <: DType](
      input: Tensor[D1],
      other: Tensor[D2]
  ): Tensor[Promoted[D1, D2]] =
    fromNative(torchNative.gammaincc(input.native, other.native))

  def parameters_to_vector(
      parameters: Seq[Tensor[?]]
  ): Tensor[?] = {
    val tensor = torchNative.parameters_to_vector(
      TensorVector(parameters.map(_.native).toArray*)
    )
    fromNative(tensor)
  }

  def vector_to_parameters(
      vec: Tensor[?],
      parameters: Seq[Tensor[?]]
  ): Unit =
    torchNative.vector_to_parameters(
      vec.native,
      TensorVector(parameters.map(_.native).toArray*)
    )
