package torch.utils.safetensors.decoder

import torch.utils.safetensors.instance.SafeTensors

/** SafeTensors loader
  */
trait ISafeTensorsLoader[B] {
  def load(tensorsDefinition: SafeTensors, blocks: B): Unit
}
