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

/** These are the basic building blocks for graphs.
  *
  * @groupname nn_conv Convolution Layers
  * @groupname nn_linear Linear Layers
  * @groupname nn_utilities Utilities
  */
package object nn {

  export modules.Module

  export modules.container.Sequential
  export modules.container.ModuleList
  export modules.container.ModuleDict
  export modules.container.Buffer
  export modules.container.Parameter
  export modules.flatten.Flatten
  export modules.linear.Linear
  export modules.linear.Identity
  export modules.linear.Bilinear
  export modules.flatten.Fold
  export modules.flatten.Unflatten
  export modules.flatten.Unfold
  export modules.normalization.GroupNorm
  export modules.normalization.LayerNorm
  export modules.normalization.RMSNorm
  export modules.normalization.LocalResponseNorm
  export modules.sparse.Embedding
  export modules.sparse.FMEmbedding

  export modules.recurrent.LSTM
  export modules.recurrent.RNNCell
  export modules.recurrent.LSTMCell
  export modules.recurrent.GRUCell
  export modules.recurrent.RNN
  export modules.recurrent.GRU
  export modules.regularization.Upsample
  export modules.regularization.Dropout
  export modules.regularization.AlphaDropout
  export modules.regularization.Dropout2d
  export modules.regularization.Dropout3d
  export modules.regularization.FeatureAlphaDropout
  export modules.sparse.EmbeddingBag
  export modules.sparse.PairwiseDistance
  export modules.sparse.CrossMapLRN2d
  export modules.sparse.CosineSimilarity
  export modules.pooling.AdaptiveAvgPool1d
  export modules.pooling.AdaptiveAvgPool2d
  export modules.pooling.AdaptiveAvgPool3d
  export modules.pooling.AdaptiveMaxPool3d
  export modules.pooling.AdaptiveMaxPool2d
  export modules.pooling.AdaptiveMaxPool1d
  export modules.pooling.AvgPool1d
  export modules.pooling.AvgPool2d
  export modules.pooling.AvgPool3d
  export modules.pooling.FractionalMaxPool2d
  export modules.pooling.FractionalMaxPool3d
  export modules.pooling.LPPool1d
  export modules.pooling.LPPool2d
  export modules.pooling.LPPool3d
  export modules.pooling.MaxPool1d
  export modules.pooling.MaxPool2d
  export modules.pooling.MaxPool3d
  export modules.pooling.MaxUnpool1d
  export modules.pooling.MaxUnpool2d
  export modules.pooling.MaxUnpool3d
  export modules.pooling.PixelShuffle
  export modules.pooling.PixelUnshuffle
  export modules.pad.ZeroPad1d
  export modules.pad.ZeroPad2d
  export modules.pad.ZeroPad3d
  export modules.pad.ConstantPad1d
  export modules.pad.ConstantPad2d
  export modules.pad.ConstantPad3d
  export modules.pad.ReflectionPad1d
  export modules.pad.ReflectionPad2d
  export modules.pad.ReflectionPad3d
  export modules.pad.ReplicationPad1d
  export modules.pad.ReplicationPad2d
  export modules.pad.ReplicationPad3d
  export modules.conv.Conv1d
  export modules.conv.Conv2d
  export modules.conv.Conv3d
  export modules.conv.ConvTranspose1d
  export modules.conv.ConvTranspose2d
  export modules.conv.ConvTranspose3d
  export modules.batchnorm.BatchNorm1d
  export modules.batchnorm.BatchNorm2d
  export modules.batchnorm.BatchNorm3d
  export modules.batchnorm.InstanceNorm1d
  export modules.batchnorm.InstanceNorm2d
  export modules.batchnorm.InstanceNorm3d
  export modules.attention.MultiheadAttention
  export modules.attention.Transformer
  export modules.attention.TransformerDecoder
  export modules.attention.TransformerDecoderLayer
  export modules.attention.TransformerEncoder
  export modules.attention.TransformerEncoderLayer
  export modules.attention.PositionalEncoding
  export modules.activation.CELU
  export modules.activation.ELU
  export modules.activation.GELU
  export modules.activation.GLU
  export modules.activation.GEGLU
  export modules.activation.Hardshrink
  export modules.activation.Hardsigmoid
  export modules.activation.Hardswish
  export modules.activation.Hardtanh
  export modules.activation.LeakyReLU
  export modules.activation.Lerp
  export modules.activation.LogSigmoid
  export modules.activation.LogSoftmax
  export modules.activation.Mish
  export modules.activation.PReLU
  export modules.activation.ReLU
  export modules.activation.ReLU6
  export modules.activation.RReLU
  export modules.activation.SELU
  export modules.activation.Sigmoid
  export modules.activation.SiLU
  export modules.activation.Softmax
  export modules.activation.Softmax2d
  export modules.activation.Softmin
  export modules.activation.Softplus
  export modules.activation.Softshrink
  export modules.activation.Softsign
  export modules.activation.Tanh
  export modules.activation.Tanhshrink
  export modules.activation.Threshold
  export loss.AdaptiveLogSoftmaxWithLoss
  export loss.BCELoss
  export loss.BCEWithLogitsLoss
  export loss.CosineEmbeddingLoss
  export loss.CrossEntropyLoss
  export loss.CTCLoss
  export loss.HingeEmbeddingLoss
  export loss.HuberLoss
  export loss.KLDivLoss
  export loss.L1Loss
  export loss.MarginRankingLoss
  export loss.MSELoss
  export loss.MultiLabelMarginLoss
  export loss.MultiLabelSoftMarginLoss
  export loss.MultiMarginLoss
  export loss.NLLLoss
  export loss.PoissonNLLLoss
  export loss.SmoothL1Loss
  export loss.SoftMarginLoss
  export loss.TripletMarginLoss
  export loss.TripletMarginWithDistanceLoss

}
