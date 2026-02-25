package torch.amp

import org.bytedeco.pytorch.global.torch
import org.bytedeco.pytorch.global.torch.DeviceType

object autocast {
  
  def is_autocast_enabled(device_type: DeviceType): Boolean ={
//    torch.set_autocast_enabled()
//    torch.is_autocast_cache_enabled()
//    torch.is_autocast_eligible()
    torch.is_autocast_enabled(device_type)
  }
//  torch.is_autocast_enabled(DeviceType.CUDA)
  
  
}
