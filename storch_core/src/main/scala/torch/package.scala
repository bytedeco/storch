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

import torch.ops

import scala.util.Using

/** The torch package contains data structures for multi-dimensional tensors and defines
  * mathematical operations over these tensors. Additionally, it provides many utilities for
  * efficient serialization of Tensors and arbitrary types, and other useful utilities.
  *
  * It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU
  * with compute capability >= 3.0.
  *
  * @groupname creation_ops Creation Ops
  * @groupname pointwise_ops Pointwise Ops
  * @groupname reduction_ops Reduction Ops
  */
package object torch
    extends ops.BLASOps
    with ops.ComparisonOps
    with ops.ContextOps
    with ops.CreationOps
    with ops.IndexingSlicingJoiningOps
    with ops.PointwiseOps
    with ops.RandomSamplingOps
    with ops.ReductionOps
    with ops.SparseOps
    with ops.JitOps
    with ops.NumpyOps
    with ops.FftOps
    with ops.GradOps
    with ops.ProfilerOPs
    with ops.UnSamplingOps
    with ops.RecurrentOps
    with ops.OtherOps {

  /** Disable gradient calculation for [[op]].
    *
    * Disabling gradient calculation is useful for inference, when you are sure that you will not
    * call `Tensor.backward()`. It will reduce memory consumption for computations that would
    * otherwise have `requiresGrad=true`.
    *
    * In this mode, the result of every computation will have `requiresGrad=false`, even when the
    * inputs have `requiresGrad=true`.
    *
    * This context manager is thread local; it will not affect computation in other threads.
    *
    * @param op
    */
  def noGrad[A](op: => A): A = {
    import org.bytedeco.pytorch.NoGradGuard
    Using.resource(NoGradGuard()) { _ =>
      op
    }
  }

  def no_grad[A](op: => A): A = {
    import org.bytedeco.pytorch.NoGradGuard
    Using.resource(NoGradGuard()) { _ =>
      op
    }
  }

  def with_grad[A](op: => A): A = {
    import org.bytedeco.pytorch.GradMode
    Using.resource(GradMode()) { _ =>
      GradMode.set_enabled(true)
      op
    }
  }

  def enable_grads[A](op: => A): A = {
    import org.bytedeco.pytorch.GradMode
    Using.resource(GradMode()) { _ =>
      GradMode.set_enabled(true)
      op
    }
  }

  def no_names_grad[A](op: => A): A = {
    import org.bytedeco.pytorch.NoNamesGuard
    Using.resource(NoNamesGuard()) { _ =>
      op
    }
  }

}
