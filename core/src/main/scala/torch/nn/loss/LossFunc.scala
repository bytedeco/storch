package torch
package nn
package loss

import torch.internal.NativeConverters.fromNative
import torch.nn.modules.Module

trait LossFunc extends Module {

  def apply[D <: DType](inputs: Tensor[D]*)(target: Tensor[?]): Tensor[D]
}
