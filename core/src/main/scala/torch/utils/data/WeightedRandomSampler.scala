package torch
package utils
package data

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{InputArchive, OutputArchive, SizeTOptional, SizeTVectorOptional}
import torch.utils.data.sampler.{Sampler, RandomSampler}

class WeightedRandomSampler[D <: FloatNN: Default](
    weights: Tensor[D],
    num_samples: Long,
    replacement: Boolean = true
) extends RandomSampler(num_samples)
    with Sampler {

  val rand_tensor = torch.multinomial(weights, num_samples, replacement)

}
