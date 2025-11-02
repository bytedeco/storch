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

import org.bytedeco.javacpp.annotation.{Const, ByRef, ByVal, Namespace}

/** @groupname nn_conv Convolution functions
  * @groupname nn_pooling Pooling functions
  * @groupname nn_attention Attention mechanisms
  * @groupname nn_activation Non-linear activation functions
  * @groupname nn_linear Linear functions
  * @groupname nn_dropout Dropout functions
  * @groupname nn_sparse Sparse functions
  * @groupname nn_distance Distance functions
  * @groupname nn_loss Loss functions
  * @groupname nn_vision Vision functions
  */
package object functional
    extends Activations
    with BatchNorm
    with Convolution
    with Distance
    with Dropout
    with Linear
    with Loss
    with Normalization
    with Linalg
    with Pooling
    with Sparse
    with UnSampling
    with Vision
    with FFT
    with Fold
    with Padding
    with Recurrent

class Functional
    extends functional.Activations
    with functional.BatchNorm
    with functional.Convolution
    with functional.Distance
    with functional.Dropout
    with functional.Linear
    with functional.Loss
    with functional.Normalization
    with functional.Linalg
    with functional.Pooling
    with functional.Sparse
    with functional.UnSampling
    with functional.Vision
    with functional.FFT
    with functional.Fold
    with functional.Padding
    with functional.Recurrent
