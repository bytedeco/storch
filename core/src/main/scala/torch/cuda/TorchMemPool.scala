package torch.cuda

import org.bytedeco.javacpp.Pointer
import org.bytedeco.pytorch.cuda.{CUDAAllocator, MemPool}

/** Scala 3包装器，用于JavaCPP的MemPool类 提供内存池管理功能，用于CUDA内存资源的优化分配
  */
class TorchMemPool private (private val native: MemPool) {

  def id = native.id()

  def allocator: CUDAAllocator = native.allocator()

  def use_count: Int = native.use_count()

  def device: Byte = native.device()

  def graph_pool_handle = MemPool.graph_pool_handle()

  def graph_pool_handle(is_user_created: Boolean = true) =
    MemPool.graph_pool_handle(is_user_created)
}
