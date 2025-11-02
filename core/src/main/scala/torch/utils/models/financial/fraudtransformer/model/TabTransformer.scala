package torch.utils.models.financial.fraudtransformer.model

import torch.*
import torch.nn.*
import torch.nn as nn
import torch.nn.modules.{HasParams, TensorModule}
import torch.optim.*
import torch.utils.data.DataLoader
import torch.utils.models.financial.fraudtransformer.layer.MultiLayerPerceptron
import torch.utils.models.financial.fraudtransformer.layer.{
  FeaturesEmbedding,
  TabTransformerEncoder
}
import scala.collection.mutable.ArrayBuffer

class TabTransformer[ParamType <: FloatNN: Default](
    catFieldDims: Array[Int],
    embedDim: Int,
    depth: Int = 2,
    nHeads: Int = 4,
    attDropout: Double = 0.5,
    ffnMult: Int = 2,
    ffnDropout: Double = 0.5,
    ffnAct: String = "GEGLU",
    anDropout: Double = 0.5,
    mlpDims: Array[Int] = Array(10, 10),
    mlpDropout: Double = 0.5
) extends TensorModule[ParamType]
    with HasParams[ParamType] {

  // 特征嵌入层
  val embedding = new FeaturesEmbedding[ParamType](catFieldDims, embedDim)
  // Transformer编码器
  val transformer = new TabTransformerEncoder[ParamType](
    inputDim = embedDim,
    depth = depth,
    nHeads = nHeads,
    attDropout = attDropout,
    ffnMult = ffnMult,
    ffnDropout = ffnDropout,
    ffnAct = ffnAct,
    anDropout = anDropout
  )

  // 嵌入输出维度计算
  val embedOutputDim = catFieldDims.length * embedDim

  // 多层感知机
  val mlp = new MultiLayerPerceptron[ParamType](
    inputDim = embedOutputDim,
    layerDims = mlpDims,
    dropout = mlpDropout
  )

  /** 前向传播
    * @param xCat
    *   类别型特征输入张量 [batchSize, numFields]
    * @return
    *   模型输出 [batchSize]
    */
  def forward(xCat: Tensor[ParamType]): Tensor[ParamType] = {
    // 1. 特征嵌入: [batchSize, numFields] -> [batchSize, numFields, embedDim]
    val embedX = embedding(xCat)
    // 2. Transformer编码: [batchSize, numFields, embedDim] -> [batchSize, numFields, embedDim]
    val transOut = transformer(embedX)
    // 3. 展平特征: [batchSize, numFields, embedDim] -> [batchSize, numFields*embedDim]
    val allX = transOut.view(transOut.size(0), -1) // 等价于PyTorch的flatten(1)
    // 4. MLP输出: [batchSize, embedOutputDim] -> [batchSize, 1] -> [batchSize]
    val out = mlp(allX)
    out.squeeze(1)
  }

  /** 应用方法，默认调用forward
    */
  override def apply(xCat: Tensor[ParamType]): Tensor[ParamType] = forward(xCat)
}

/** TabTransformer 训练工具
  */
object TabTransformerTrainer {

  /** 训练模型
    * @param model
    *   待训练的TabTransformer模型
    * @param trainData
    *   训练数据集 (输入, 标签)
    * @param epochs
    *   训练轮数
    * @param lr
    *   学习率
    * @tparam ParamType
    *   数值类型
    */
//  def train[ParamType <: FloatNN: Default](
//      model: TabTransformer[ParamType],
//      trainData: DataLoader[(Tensor[ParamType], Tensor[ParamType])],
//      epochs: Int,
//      lr: Double = 0.001
//  ): Unit = {
//    // 优化器
//    val optimizer = Adam(model.parameters(true), lr)
//    // 损失函数 (根据任务可替换为CrossEntropyLoss等)
//    val criterion = MSELoss()
//    for (epoch <- 1 to epochs) {
//      var totalLoss = 0.0
//      var batchCount = 0
//      model.train()
//      for ((x, y) <- trainData) {
//        // 梯度清零
//        optimizer.zeroGrad()
//        // 前向传播
//        val output = model(x)
//        // 计算损失
//        val loss = criterion(output, y)
//        // 反向传播
//        loss.backward()
//        // 参数更新
//        optimizer.step()
//        // 累计损失
//        totalLoss += loss.item()
//        batchCount += 1
//      }
//      // 打印 epoch 信息
//      println(f"Epoch $epoch%3d, Average Loss: ${totalLoss / batchCount}%.6f")
//    }
//  }
}
