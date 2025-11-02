package torch
package utils
package data

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional}
import torch.utils.data.sampler.{Sampler, RandomSampler}

class SubsetRandomSampler(indices: Seq[Long]) extends RandomSampler(indices.size) with Sampler {

  val len = indices.size
  val rand_tensor = torch.randperm(len).tolist()

}
