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
package optim

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{
  LBFGSOptions,
  LBFGSParamState,
  OptimizerParamState,
  LongOptional,
  StringOptional,
  TensorVector
}

import scala.collection.immutable.Iterable

// format: off
/** Implements the AdamW algorithm.
 *torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None,
 * tolerance_grad=1e-07,
 * tolerance_change=1e-09, history_size=100, line_search_fn=None)
 */
// format: on
final class LBFGS(
    params: Iterable[Tensor[?]],
    lr: Double = 1e-3,
    maxIter: Long = 20,
    maxEval: Option[Long] = None,
    toleranceGrad: Double = 0.9,
    toleranceChange: Double = 1e-8,
    historySize: Long = 0,
    lineSearchFn: Option[String] = None
) extends Optimizer {
  private val nativeParams: TensorVector = TensorVector(params.map(_.native).toArray*)
  private val options: LBFGSOptions = LBFGSOptions(lr)

  options.max_iter().put(maxIter)

  if (maxEval.isDefined) {
    val nativeMaxEval = new LongOptional(maxEval.get)
    options.max_eval().put(nativeMaxEval)
  }
  options.tolerance_grad().put(toleranceGrad)
  options.tolerance_change().put(toleranceChange)
  options.history_size().put(historySize)

  if (lineSearchFn.isDefined) {
    val nativeLineSearchFn = new StringOptional(lineSearchFn.get)
    options.line_search_fn().put(nativeLineSearchFn) // StringOptional
  }
  override val optimizerParamState: OptimizerParamState = new LBFGSParamState()
  override private[torch] val native: pytorch.LBFGS = pytorch.LBFGS(nativeParams, options)
}

object LBFGS:
  def apply(
      params: Iterable[Tensor[?]],
      lr: Double = 1e-3,
      max_iter: Long = 20,
      max_eval: Option[Long] = None,
      tolerance_grad: Double = 0.999,
      tolerance_change: Double = 1e-8,
      history_size: Long = 0,
      line_search_fn: Option[String] = None
  ): LBFGS = new LBFGS(
    params,
    lr,
    max_iter,
    max_eval,
    tolerance_grad,
    tolerance_change,
    history_size,
    line_search_fn
  )
