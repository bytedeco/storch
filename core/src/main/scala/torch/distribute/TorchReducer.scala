package torch
package distribute

import org.bytedeco.pytorch.global.torch.{BuiltinCommHookType, ScalarType}
import org.bytedeco.pytorch.{
  Reducer,
  TensorVector,
  SizeTStringMap,
  Future,
//  LongArrayRefVector,
//  LongArrayRef,
  Work,
//  LongVector,
  SizeTVector,
  SizeTVectorVector,
  BoolVector,
  CommHookInterface,
//  TensorOptional,
  GradBucket,
  StringTensorMap,
  FutureList,
  Logger,
  ProcessGroup
}
import torch.internal.NativeConverters.fromNative

import scala.collection.mutable.ListBuffer
abstract class TorchReducer {

  val native: Reducer

  def getReducer(
      params: Array[Tensor[?]],
      bucket_indices: Array[Long],
      pg: ProcessGroup,
      expect_sparse_gradients: Array[Boolean],
      bucket_bytes_cap: Long,
      find_unused_parameters: Boolean,
      gradient_as_bucket_view: Boolean,
      param_names: SizeTStringMap,
      first_bucket_bytes_cap: Long
  ): Reducer = {
    val paramsNative = new TensorVector(params.map(_.native)*)
    val buckets = new SizeTVectorVector(bucket_indices.map(el => new SizeTVector(el))*)
    val expectGradients = new BoolVector(expect_sparse_gradients*)
    new Reducer(
      paramsNative,
      buckets,
      pg,
      expectGradients,
      bucket_bytes_cap,
      find_unused_parameters,
      gradient_as_bucket_view,
      param_names,
      first_bucket_bytes_cap
    )
  }

  def install_futures(futs: FutureList) = native.install_futures(futs)

  def set_forward_pass_work_handle(forwardPassWorkHandle: Work, useStaticWorldSize: Boolean) =
    native.set_forward_pass_work_handle(forwardPassWorkHandle, useStaticWorldSize)

//  def setSparseMetadata(metadata: Map[String, Tensor[?]]) = {
//    native.setSparseMetadata()
//  }

  def setSparseMetadata(metadata: StringTensorMap) = {
    native.setSparseMetadata(metadata)
  }

//  def prepare_for_forward(tensors: Array[Tensor[?]]) = {
//    val vec = new TensorVector(tensors.map(_.native)*)
//    native.prepare_for_forward(vec)
//  }

//  def set_mixed_precision_param_dtype(dtype: DType) = native.set_mixed_precision_param_dtype(Default.scalaToDType(dtype))

  def set_mixed_precision_param_dtype(dtype: ScalarType) =
    native.set_mixed_precision_param_dtype(dtype)

  //  public native void initialize_buckets(@ByVal SizeTVectorVector bucket_indices);
  def initialize_buckets(bucket_indices: Array[Long]) =
    native.initialize_buckets(new SizeTVectorVector(bucket_indices.map(el => new SizeTVector(el))*))

  def autograd_hook(index: Long) = native.autograd_hook(index)

  def prepare_for_forward = native.prepare_for_forward()

  def get_backward_stats: Seq[Long] = {

    val vector = native.get_backward_stats()
    val buffer = new ListBuffer[Long]()
    var it = vector.begin()
    while (!it.equals(vector.end())) {
      buffer.append(it.get())
      it = it.increment()
    }
    buffer.toSeq

  }

  def register_comm_hook(interface: CommHookInterface) = native.register_comm_hook(interface)

  def register_builtin_comm_hook(comm_hook_type: BuiltinCommHookType) =
    native.register_builtin_comm_hook(comm_hook_type)

  def register_builtin_comm_hook(comm_hook_type: Byte) =
    native.register_builtin_comm_hook(comm_hook_type)

  def set_optimizer_in_backward = native.set_optimizer_in_backward()

  def run_comm_hook(grad_bucket: GradBucket): Future = native.run_comm_hook(grad_bucket)

  def run_allreduce_hook(grad_bucket: GradBucket): Future = native.run_allreduce_hook(grad_bucket)

  def get_grad_buckets: GradBucket = native.get_grad_buckets()

  def get_grad_buckets(return_zero_tensors: Boolean): GradBucket =
    native.get_grad_buckets(return_zero_tensors)

  def rebuild_buckets: Boolean = native.rebuild_buckets()

  def should_rebuild_buckets: Boolean = native.should_rebuild_buckets()

  def push_rebuilt_params_for_all_indices = native.push_rebuilt_params_for_all_indices()

  def get_local_used_map_on_device = fromNative(native.get_local_used_map_on_device())

  def set_logger(logger: Logger) = native.set_logger(logger)
  def set_ddp_runtime_logging_sample_rate(sample_rate: Int) =
    native.set_ddp_runtime_logging_sample_rate(sample_rate)

  def set_static_graph = native.set_static_graph()

  def delay_all_reduce = native.delay_all_reduce()

  def ddp_graph_static = native.ddp_graph_static()

  def remove_autograd_hooks = native.remove_autograd_hooks()

  def check_finalized = native.check_finalized

  def reset_state = native.reset_state()

  def update_process_group(new_process_group: ProcessGroup) =
    native.update_process_group(new_process_group)

}
