//package torch.nn.loss //AdaptiveLogSoftmaxWithLossImpl
package torch
package nn
package loss

import torch.nn.modules.Module
import torch.internal.NativeConverters.fromNative
import org.bytedeco.pytorch.{
  AdaptiveLogSoftmaxWithLossImpl,
  AdaptiveLogSoftmaxWithLossOptions,
  LongVector
}

//class torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, device=None, dtype=None)[source]
final class AdaptiveLogSoftmaxWithLoss(
    val inFeatures: Long,
    val nClasses: Long,
    val cutoffs: Array[Long],
    val divValue: Double = 4.0,
    val headBias: Boolean = false
) extends LossFunc {

  val cutoffsVec = LongVector(cutoffs*)
  val options = new AdaptiveLogSoftmaxWithLossOptions(inFeatures, nClasses, cutoffsVec)
  options.div_value.put(divValue)
  options.head_bias.put(headBias)
  options.n_classes.put(nClasses)
  options.in_features.put(inFeatures)
  options.cutoffs().put(cutoffsVec)

  override private[torch] val nativeModule: AdaptiveLogSoftmaxWithLossImpl =
    AdaptiveLogSoftmaxWithLossImpl(options)

  override def hasBias(): Boolean = false

  def reset(): Unit = nativeModule.reset()

  def head_size(): Long = nativeModule.head_size()

  def n_clusters(): Long = nativeModule.n_clusters()

  def shortlist_size(): Long = nativeModule.shortlist_size()

  def predict[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.predict(input.native)
  )

  def cutoffs_() = nativeModule.cutoffs() // todo make as Seq[Long]

  def log_prob[D <: DType](input: Tensor[D]): Tensor[D] = fromNative(
    nativeModule.log_prob(input.native)
  )

  def get_full_log_prob[D <: DType](input: Tensor[D], head_output: Tensor[D]): Tensor[D] =
    fromNative(nativeModule._get_full_log_prob(input.native, head_output.native))

  def apply[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = fromNative(
    nativeModule.forward(input.native, target.native).output()
  )
  def forward[D <: DType](input: Tensor[D], target: Tensor[?]): Tensor[D] = apply(input, target)

  override def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native).output()
    )
  }

  def forward[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D] = {
    val input = inputs.toSeq.head
    fromNative(
      nativeModule.forward(input.native, target.native).output()
    )
  }

}

object AdaptiveLogSoftmaxWithLoss:
  def apply(
      in_features: Long,
      n_classes: Long,
      cutoffs: Array[Long],
      div_value: Double = 4.0,
      head_bias: Boolean = false
  ): AdaptiveLogSoftmaxWithLoss =
    new AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value, head_bias)

//  /** Given input tensor, and output of {@code head}, computes the log of the full
//   *  distribution */

//
//  /** Computes log probabilities for all n_classes */
//
//  /** This is equivalent to {@code log_pob(input).argmax(1)} but is more efficient in
//   *  some cases */
//
//  /** The options with which this {@code Module} was constructed */
//
//  /** Cutoffs used to assign targets to their buckets. It should be an ordered
//   *  Sequence of integers sorted in the increasing order */
//
//
//  /** Number of clusters */
//
//  /** Output size of head classifier */
