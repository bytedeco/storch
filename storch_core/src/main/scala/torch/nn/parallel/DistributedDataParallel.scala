package torch
package nn
package parallel

object DistributedDataParallel {

  println(s"DistributedDataParallel is available: ${torch.cuda.is_available()}")
}
