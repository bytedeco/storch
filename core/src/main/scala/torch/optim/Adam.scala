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
import org.bytedeco.pytorch.{AdamOptions, AdamParamState, OptimizerParamState, TensorVector}

import scala.collection.immutable.Iterable

// format: off
/** Implements the Adam algorithm.
 *
 * $$
 * \begin{aligned}
 *      &\rule{110mm}{0.4pt}                                                                 \\
 *      &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
 *          \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
 *      &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
 *          \:\textit{maximize}                                                              \\
 *      &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
 *          v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
 *      &\rule{110mm}{0.4pt}                                                                 \\
 *      &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
 * 
 *      &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
 *      &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
 *      &\hspace{5mm}\textbf{else}                                                           \\
 *      &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
 *      &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
 *      &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
 *      &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
 *      &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
 *      &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
 *      &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
 *      &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
 *      &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
 *          \widehat{v_t})                                                                   \\
 *      &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
 *          \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
 *      &\hspace{5mm}\textbf{else}                                                           \\
 *      &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
 *  \end{aligned}
 *  $$
 *
 *  For further details regarding the algorithm we refer to
 *  [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
//  torch.optim.SGD：随机梯度下降优化器
//
//2. torch.optim.Adam：Adam优化器
//
//3. torch.optim.Adadelta：Adadelta优化器
//
//4. torch.optim.Adagrad：Adagrad优化器
//
//5. torch.optim.Adamax：Adamax优化器
//
//6. torch.optim.ASGD：平均随机梯度下降优化器
//
//7. torch.optim.RMSprop：RMSprop优化器
//
//8. torch.optim.Rprop：Rprop优化器
// 
//9. torch.optim.LBFGS：L-BFGS优化器
 torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
 *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
// format: on */

final class Adam(
    params: Iterable[Tensor[?]],
    lr: Double = 1e-3,
    betas: (Double, Double) = (0.9, 0.999),
    eps: Double = 1e-8,
    weightDecay: Double = 0,
    amsgrad: Boolean = false
) extends Optimizer {
  private val nativeParams: TensorVector = TensorVector(params.map(_.native).toArray*)
  private val options: AdamOptions = AdamOptions(lr)
  options.betas().put(Array(betas._1, betas._2)*)
  options.eps().put(eps)
  options.weight_decay().put(weightDecay)
  options.amsgrad().put(amsgrad)
  override val optimizerParamState: OptimizerParamState = new AdamParamState()
  override private[torch] val native: pytorch.Adam = pytorch.Adam(nativeParams, options)
}

object Adam:
  def apply(
             params: Iterable[Tensor[?]],
             lr: Double = 1e-3,
             betas: (Double, Double) = (0.9, 0.999),
             eps: Double = 1e-8,
             weight_decay: Double = 0,
             amsgrad: Boolean = false
           ) :Adam = new Adam(params,lr,betas,eps,weight_decay,amsgrad)