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
package loss

import org.bytedeco.pytorch.{
  CrossEntropyLossImpl,
  CrossEntropyLossOptions,
  LossReduction,
  kMean,
  kSum,
  kNone
}
import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

/** This criterion computes the cross entropy loss between input and target. */
// TODO optional args
//class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)[source]
final class CrossEntropyLoss(
    val weight: Option[Tensor[?]] = None,
    val ignore_index: Long = -100,
    val reduction: String = "mean",
    val label_smoothing: Double = 0.0,
    val size_average: Option[Boolean] = None,
    val reduce: Option[Boolean] = None
) extends LossFunc {

  private[torch] val options: CrossEntropyLossOptions = new CrossEntropyLossOptions()
  val lossReduction = reduction match {
    case "mean" | "Mean" | "MEAN" => new LossReduction(new kMean())
    case "sum" | "Sum" | "SUM"    => new LossReduction(new kSum())
    case "none" | "None" | "NONE" => new LossReduction(new kNone())
    case _ => throw new IllegalArgumentException(s"Unknown reduction $reduction")
  }
  options.reduction().put(lossReduction)
  options.ignore_index().put(ignore_index)
  options.label_smoothing().put(label_smoothing)
  if weight.isDefined then options.weight().put(weight.get.native)

  override private[torch] val nativeModule: CrossEntropyLossImpl = CrossEntropyLossImpl(options)

  def weight[D <: DType](): Tensor[D] = fromNative(nativeModule.weight())
  def reset(): Unit = nativeModule.reset()
  override def hasBias(): Boolean = false

  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native)
  )

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }

  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native) // .output()
    )
  }
}

object CrossEntropyLoss {
  def apply(
      weight: Option[Tensor[?]] = None,
      ignore_index: Long = -100,
      reduction: String = "mean",
      label_smoothing: Double = 0.0,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): CrossEntropyLoss =
    new CrossEntropyLoss(weight, ignore_index, reduction, label_smoothing, size_average, reduce)
}
