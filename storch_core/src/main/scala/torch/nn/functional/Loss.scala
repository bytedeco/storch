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
package functional

import org.bytedeco.pytorch.{
  NLLLossOptions,
  MSELossOptions,
  KLDivLossReduction,
  L1LossOptions,
  KLDivLossOptions,
  kBatchMean,
  PoissonNLLLossOptions,
  kSum,
  kMean,
  kNone,
  LossReduction,
  BCEWithLogitsLossOptions,
  BCELossOptions,
  HingeEmbeddingLossOptions,
  MultiMarginLossOptions,
  CosineEmbeddingLossOptions,
  SmoothL1LossOptions,
  HuberLossOptions,
  SoftMarginLossOptions,
  MultiLabelMarginLossOptions,
  MultiLabelSoftMarginLossOptions,
  TripletMarginLossOptions,
  TripletMarginWithDistanceLossOptions,
  CTCLossOptions,
  TensorOptional,
  MarginRankingLossOptions,
  CrossEntropyLossOptions
}
import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative

// Loss functions
private[torch] trait Loss {

  /** Function that measures Binary Cross Entropy between target and input logits.
    *
    * TODO support weight, reduction, pos_weight
    *
    * @group nn_loss
    */
  def binary_cross_entropy_with_logits[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = binaryCrossEntropyWithLogits(input, target)

  def binaryCrossEntropyWithLogits[
      I <: BFloat16 | Float32 | Float64,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.binary_cross_entropy_with_logits(
        input.native,
        target.native,
        BCEWithLogitsLossOptions()
      )
    )
  }

  // format: off
  // http://bytedeco.org/javacpp-presets/pytorch/apidocs/

  /** This criterion computes the cross entropy loss between input logits and target. See
   * [[torch.nn.loss.CrossEntropyLoss]] for details.
   *
   * **Shape:**
   *
   * * Input: Shape $(C)$, $(N,C)$ or $(N,C,d_1,d_2,...,d_K)$ with $K≥1$ in the case of K-dimensional
   * loss.
   * * Target: If containing class indices, shape $()$, $(N)$ or $(N,d_1,d_2,...,d_K)$ with $K≥1$ 
   * in the case of K-dimensional loss where each value should be between $[0,C)$. If containing class 
   * probabilities, same shape as the input and each value should be between [0,1][0,1].
   *
   * where:
   * * C = number of classes
   * * N = batch size​
   *
   * @example
   * ```scala
   * // Example of target with class indices
   * val input = torch.randn(3, 5, requires_grad=True)
   * val target = torch.randint(5, (3,), dtype=torch.int64)
   * val loss = F.cross_entropy(input, target)
   * loss.backward()
   *
   * // Example of target with class probabilities
   * val input = torch.randn(3, 5, requires_grad=True)
   * val target = torch.randn(3, 5).softmax(dim=1)
   * val loss = F.crossEntropy(input, target)
   * loss.backward()
   * ```
   *
   * @param input
   * Predicted unnormalized logits; see Shape section above for supported shapes.
   * @param target
   * Ground truth class indices or class probabilities; see Shape section below for supported
   * shapes.
   * @param weight
   * a manual rescaling weight given to each class. If given, has to be a Tensor of size C
   * @param size_average
   * Deprecated (see reduction). By default, the losses are averaged over each loss element in
   * the batch. Note that for some losses, there multiple elements per sample. If the field
   * `size_average` is set to `false`, the losses are instead summed for each mini-batch. Ignored
   * when reduce is `false`. Default: `true`
   * @param ignore_index
   * Specifies a target value that is ignored and does not contribute to the input gradient. When
   * `size_average` is `true`, the loss is averaged over non-ignored targets. Note that
   * `ignore_index` is only applicable when the target contains class indices. Default: `-100`
   * @param reduce
   * Deprecated (see reduction). By default, the losses are averaged or summed over observations
   * for each mini-batch depending on `size_average`. When reduce is `false`, returns a loss per
   * batch element instead and ignores size_average. Default: `true`
   * @param reduction
   * Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no
   * reduction will be applied, 'mean': the sum of the output will be divided by the number of
   * elements in the output, 'sum': the output will be summed. Note: `size_average` and `reduce`
   * are in the process of being deprecated, and in the meantime, specifying either of those two
   * args will override reduction. Default: 'mean'
   * @param label_smoothing
   * A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0
   * means no smoothing. The targets become a mixture of the original ground truth and a uniform
   * distribution as described in
   * [[https://arxiv.org/abs/1512.00567 Rethinking the Inception Architecture for Computer Vision]].
   * Default: 0.0
   * @return
   * [[torch.Tensor]]
   * @see
   * See
   * [[https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html torch.nn.functional.cross_entropy]]
   * @see
   * See
   * [[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for equivalent torch.nn.CrossEntropyLoss class]]
   * @see
   * See [[https://pytorch.org/cppdocs/ PyTorch C++ documentation]]
   * @see
   * See [[http://bytedeco.org/javacpp-presets/pytorch/apidocs/ ByteDeco PyTorch preset]]
   */
  // format: on

  def cross_entropy[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN // not be Int , must long
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[I] =
    fromNative(
      torchNative.cross_entropy(
        input.native,
        target.to(dtype = torch.int64).native
      )
    )

  def cross_entropy[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN // not be Int , must long
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[I],
      label_smoothing: Double,
      ignore_index: Long,
      reduction: String // = "mean"
  ): Tensor[I] = {
    val options = CrossEntropyLossOptions()
    options.label_smoothing().put(label_smoothing)
    options.ignore_index().put(ignore_index)
    options.weight().put(weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }

    options.reduction().put(LossReduction(reductionNative))

    fromNative(
      torchNative.cross_entropy(
        input.native,
        target.to(dtype = torch.int64).native,
        options
      )
    )

  }

  def binary_cross_entropy_with_logits[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      pos_weight: Tensor[O],
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = BCEWithLogitsLossOptions()
    options.weight().put(weight.native)
    options.pos_weight().put(pos_weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(reductionNative))

    fromNative(torchNative.binary_cross_entropy_with_logits(input.native, target.native, options))
  }

  def binary_cross_entropy[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.binary_cross_entropy(
        input.native,
        target.native
      )
    )
  }
  def binary_cross_entropy[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {

    val options = BCELossOptions()
    options.weight().put(weight.native)
    val reductionNative = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(reductionNative))
    fromNative(
      torchNative.binary_cross_entropy(input.native, target.to(dtype = torch.int64).native, options)
    )
  }

  // torch.nn.functional.ctc_loss(log_probs, targets,
  // input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)[source]

  def ctc_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      input_lengths: Tensor[Int64],
      target_lengths: Tensor[Int64]
  ): Tensor[O] = {
    fromNative(
      torchNative.ctc_loss(
        input.native,
        target.native,
        input_lengths.native,
        target_lengths.native
      )
    )
  }
  def ctc_loss[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      log_probs: Tensor[I],
      targets: Tensor[O],
      input_lengths: Tensor[Int64],
      target_lengths: Tensor[Int64],
      blank: Int = 0,
      reduction: String = "mean",
      zero_infinity: Boolean = false
  ): Tensor[O] = {
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    val options = new CTCLossOptions()
    options.blank().put(blank.toLong)
    options.zero_infinity().put(zero_infinity)
    options.reduction().put(LossReduction(nativeReduction))
    fromNative(
      torchNative.ctc_loss(
        log_probs.native,
        targets.native,
        input_lengths.native,
        target_lengths.native,
        options
      )
    )
  }

  def poisson_nll_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.poisson_nll_loss(
        input.native,
        target.native
      )
    )
  }
  def poisson_nll_loss[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      log_input: Boolean,
      full: Boolean,
      eps: Float,
      reduction: String // mean
  ): Tensor[O] = {
    val options = PoissonNLLLossOptions()
    options.log_input().put(log_input)
    options.full().put(full)
    options.eps().put(eps)
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))
    fromNative(
      torchNative.poisson_nll_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def cosine_embedding_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input1: Tensor[I],
      input2: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.cosine_embedding_loss(
        input1.native,
        input2.native,
        target.native
      )
    )
  }
  def cosine_embedding_loss[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input1: Tensor[I],
      input2: Tensor[I],
      target: Tensor[O],
      margin: Float,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = CosineEmbeddingLossOptions()
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.cosine_embedding_loss(
        input1.native,
        input2.native,
        target.native,
        options
      )
    )

  }

  def cross_entropy[
      I <: BFloat16 | Float32 | Float64,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      ignore_index: Long,
      label_smoothing: Float,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = CrossEntropyLossOptions()
    options.weight().put(weight.native)
    options.ignore_index().put(ignore_index)
    options.label_smoothing().put(label_smoothing)
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(torchNative.cross_entropy(input.native, target.native, options))

  }

  //  def ctcLoss[]

  //  def gaussianNllLoss[D <:DType](
  //      input: Tensor[D],
  //      target: Tensor[D],
  //      p: Double,
  //      sizeAverage: Boolean,
  //      reduce: Boolean,
  //      reduction: String
  //  ): Tensor[D] =
  //
  //    fromNative(
  //    torchNative.gaussian_nll_loss(
  //      input.native,
  //      target.native,
  //      p,
  //      sizeAverage,
  //      reduce,
  //      reduction
  //    )
  //  )
  def hinge_embedding_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.hinge_embedding_loss(
        input.native,
        target.native
      )
    )
  }
  def hinge_embedding_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      margin: Double,
      p: Double,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {

    val options = HingeEmbeddingLossOptions()
    options.margin().put(margin)
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))
    fromNative(
      torchNative.hinge_embedding_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def kl_div[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.kl_div(
        input.native,
        target.native
      )
    )
  }
  def kl_div[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      log_target: Boolean,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {

    val options = KLDivLossOptions()
    options.log_target().put(log_target)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN"                                            => new kMean()
      case "sum" | "Sum" | "SUM"                                               => new kSum()
      case "none" | "None" | "NONE"                                            => new kNone()
      case "batchMean" | "BatchMean" | "batchmean" | "BATCHMEAN" | "BatchMean" => new kBatchMean()
    }
    options.reduction().put(KLDivLossReduction(nativeReduction))

    fromNative(
      torchNative.kl_div(
        input.native,
        target.native,
        options
      )
    )

  }

  def l1_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.l1_loss(
        input.native,
        target.native
      )
    )
  }
  def l1_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = L1LossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.l1_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def mse_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.mse_loss(
        input.native,
        target.native
      )
    )
  }
  def mse_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = MSELossOptions()
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.mse_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def margin_ranking_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input1: Tensor[I],
      input2: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.margin_ranking_loss(
        input1.native,
        input2.native,
        target.native
      )
    )
  }
  def margin_ranking_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input1: Tensor[I],
      input2: Tensor[I],
      target: Tensor[O],
      margin: Double,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = MarginRankingLossOptions()
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.margin_ranking_loss(
        input1.native,
        input2.native,
        target.native,
        options
      )
    )

  }

  def multilabel_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.multilabel_margin_loss(
        input.native,
        target.native
      )
    )
  }
  def multilabel_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      margin: Double,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = MultiLabelMarginLossOptions()
    //    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.multilabel_margin_loss(
        input.native,
        target.native,
        options
      )
    )

  }

  def multilabel_soft_margin_loss[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.multilabel_soft_margin_loss(
        input.native,
        target.native
      )
    )
  }
  def multilabel_soft_margin_loss[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      margin: Double,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = MultiLabelSoftMarginLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.weight().put(weight.native)
    fromNative(
      torchNative.multilabel_soft_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def multi_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.multi_margin_loss(
        input.native,
        target.native
      )
    )
  }
  def multi_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      p: Double,
      margin: Double,
      weight: Tensor[O],
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = MultiMarginLossOptions()
    options.p().put(p.toLong)
    options.margin().put(margin)
    options.weight().put(weight.native)
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.multi_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def nll_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      weight: Tensor[O],
      reduction: String = "mean",
      ignore_index: Int,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = NLLLossOptions()
    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }

    options.reduction().put(LossReduction(nativeReduction))
    options.weight().put(weight.native)
    options.ignore_index().put(ignore_index.toLong)
    fromNative(
      torchNative.nll_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def nll_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.nll_loss(
        input.native,
        target.native
      )
    )
  }
//long reduction/*=at::Reduction::Mean*/, @Cast("int64_t") long ignore_index/*=-100*/
  def nll_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      weight: Option[Tensor[O]],
      reduction: Int,
      ignore_index: Int
  ): Tensor[O] = {
//    val nativeReduction = reduction match {
//      case "none" | "None" | "NONE" => 0l //new kNone()
//      case "mean" | "Mean" | "MEAN" => 1l //new kMean()
//      case "sum" | "Sum" | "SUM" => 2l //new kSum()
//      case "end" | "End" | "END" => 3l //new kMean()
//    }
    val nativeWeight = if (weight.isDefined) TensorOptional(weight.get.native) else TensorOptional()
    fromNative(
      torchNative.nll_loss(
        input.native,
        target.native,
        nativeWeight,
        reduction.toLong,
        ignore_index.toLong
      )
    )
  }

  def nll_loss2d[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.nll_loss2d(
        input.native,
        target.native
      )
    )
  }

  // (@Const @ByRef Tensor self, @Const @ByRef Tensor target,
  // TensorOptional weight,
  // long reduction/*=at::Reduction::Mean*/,
  // long ignore_index/*=-100*/)
  // @Namespace("at::Reduction") public enum Reduction {
  //  None(0), // Do not reduce
  //  Mean(1), // (Possibly weighted) mean of losses
  //  Sum(2), // Sum losses
  //  END(3);
  def nll_loss2d[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      weight: Option[Tensor[O]] = None,
      reduction: String = "mean",
      ignore_index: Int = 100,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {

    val nativeReduction = reduction match {
      case "none" | "None" | "NONE" => 0L // new kNone()
      case "mean" | "Mean" | "MEAN" => 1L // new kMean()
      case "sum" | "Sum" | "SUM"    => 2L // new kSum()
      case "end" | "End" | "END"    => 3L // new kMean()
    }
    //  options.reduction().put(nativeReduction)
    val nativeWeight = if (weight.isDefined) TensorOptional(weight.get.native) else TensorOptional()
    fromNative(
      torchNative.nll_loss2d(
        input.native,
        target.native,
        nativeWeight,
        nativeReduction,
        ignore_index.toLong
      )
    )
  }

//  def poisson_nll_loss[D <: DType](
//                              input: Tensor[D],
//                              target: Tensor[D],
//                            ): Tensor[D] = {
//    fromNative(
//      torchNative.poisson_nll_loss(
//        input.native,
//        target.native
//      )
//    )
//  }
  def poisson_nll_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      logInput: Boolean,
      full: Boolean,
      eps: Double,
      reduction: String,
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = PoissonNLLLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.eps().put(eps)
    options.full().put(full)
    options.log_input().put(logInput)
    fromNative(
      torchNative.poisson_nll_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def huber_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.huber_loss(
        input.native,
        target.native
      )
    )
  }
  def huber_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      delta: Double,
      reduction: String = "mean"
  ): Tensor[O] = {
    val options = HuberLossOptions()
    options.delta().put(delta)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.huber_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def smooth_l1_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.smooth_l1_loss(
        input.native,
        target.native
      )
    )
  }
  def smooth_l1_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      beta: Double,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = SmoothL1LossOptions()
    options.beta().put(beta)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.smooth_l1_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def soft_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.soft_margin_loss(
        input.native,
        target.native
      )
    )
  }
  def soft_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      input: Tensor[I],
      target: Tensor[O],
      reduction: String = "mean"
  ): Tensor[O] = {
    val options = SoftMarginLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.soft_margin_loss(
        input.native,
        target.native,
        options
      )
    )
  }

  def triplet_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      anchor: Tensor[I],
      positive: Tensor[O],
      negative: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.triplet_margin_loss(
        anchor.native,
        positive.native,
        negative.native
      )
    )
  }
  def triplet_margin_loss[I <: BFloat16 | FloatNN, O <: NumericRealNN](
      anchor: Tensor[I],
      positive: Tensor[O],
      negative: Tensor[O],
      margin: Double,
      p: Double,
      eps: Double,
      swap: Boolean,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = TripletMarginLossOptions()
    options.p().put(p)
    options.eps().put(eps)
    options.swap().put(swap)
    options.margin().put(margin)

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))

    fromNative(
      torchNative.triplet_margin_loss(anchor.native, positive.native, negative.native, options)
    )
  }

  def triplet_margin_with_distance_loss[
      I <: BFloat16 | FloatNN,
      O <: NumericRealNN
  ](
      anchor: Tensor[I],
      positive: Tensor[O],
      negative: Tensor[O]
  ): Tensor[O] = {
    fromNative(
      torchNative.triplet_margin_with_distance_loss(
        anchor.native,
        positive.native,
        negative.native
      )
    )
  }
  def tripletMarginWithDistanceLoss[
      I <: BFloat16 | FloatNN,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      anchor: Tensor[I],
      positive: Tensor[O],
      negative: Tensor[O],
      margin: Double,
      p: Double,
      eps: Double,
      swap: Boolean,
      distanceFunction: String,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = triplet_margin_with_distance_loss(
    anchor,
    positive,
    negative,
    margin,
    p,
    eps,
    swap,
    distanceFunction,
    reduction,
    size_average,
    reduce
  )

  def triplet_margin_with_distance_loss[
      I <: BFloat16 | FloatNN,
      O <: BFloat16 | Float16 | Float32 | Float64
  ](
      anchor: Tensor[I],
      positive: Tensor[O],
      negative: Tensor[O],
      margin: Double,
      p: Double,
      eps: Double,
      swap: Boolean,
      distance_function: String,
      reduction: String = "mean",
      size_average: Option[Boolean] = None,
      reduce: Option[Boolean] = None
  ): Tensor[O] = {
    val options = TripletMarginWithDistanceLossOptions()

    val nativeReduction = reduction match {
      case "mean" | "Mean" | "MEAN" => new kMean()
      case "sum" | "Sum" | "SUM"    => new kSum()
      case "none" | "None" | "NONE" => new kNone()
    }
    options.reduction().put(LossReduction(nativeReduction))
    options.swap().put(swap)
    options.margin().put(margin)
//      options.distance_function().put(Pointer(distanceFunction))

    fromNative(
      torchNative.triplet_margin_with_distance_loss(
        anchor.native,
        positive.native,
        negative.native,
        options
      )
    )
  }
}
