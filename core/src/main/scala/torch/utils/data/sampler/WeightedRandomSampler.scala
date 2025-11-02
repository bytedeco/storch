package torch.utils.data.sampler

//package torch.data.sampler
//
//import org.bytedeco.pytorch
//import org.bytedeco.pytorch.{RandomSampler,InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional, T_TensorT_TensorTensor_T_T, T_TensorTensor_T, T_TensorTensor_TOptional, TensorMapper, TensorVector, TransformerImpl, TransformerOptions, kCircular, kGELU, kReflect, kReplicate, kZeros, RandomSampler as RS}
//import torch.utils.data.dataset.Dataset
//import torch.internal.NativeConverters.{fromNative, toNative}
//
//class WeightedRandomSampler (dataset:Dataset) extends RS with Sampler {
//
//  override def reset(new_size: SizeTOptional): Unit = super.reset(new_size)
//
//  override def reset(): Unit = super.reset()
//
//  override def next(batch_size: Long): SizeTVectorOptional = super.next(batch_size)
//
//  override def save(archive: OutputArchive): Unit = super.save(archive)
//
//  override def load(archive: InputArchive): Unit = super.load(archive)
//
//  override def index(): Long = super.index()
//}
//
//
//
