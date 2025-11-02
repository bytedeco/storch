package torch.utils.models.llm.gpt

import torch.*
import torch.nn.{Module, functional as F}
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.linear.Linear
import torch.nn.modules.sparse.Embedding
import torch.nn.modules.{HasParams, TensorModule}

class GPT[ParamType <: FloatNN: Default](config: GPTConfig) extends HasParams[ParamType] {

  val tokenEmbeddingTable = register(nn.Embedding(config.vocab_size, config.n_embd))
  val positionEmbeddingTable = register(nn.Embedding(config.block_size, config.n_embd))
  val blocks = nn.Sequential[ParamType](
    (0 until config.n_layer).map(_ => new Block[ParamType](config)): _*
  )
  val lnFinal = register(nn.LayerNorm[ParamType](Seq(config.n_embd)))
  val lmHead = register(nn.Linear[ParamType](config.n_embd, config.vocab_size, bias = false))
  initWeights()

  private def initWeights(): Unit = {
    // initialize the weights of all the submodules
    def initModule(module: Module): Unit = {
      module match {
        case linear: Linear[ParamType] =>
          torch.nn.init.normal_(linear.weight, 0.0f, 0.02f)
          if (linear.hasBias()) torch.nn.init.zeros_(linear.bias)
        case embedding: Embedding[ParamType] =>
          torch.nn.init.normal_(embedding.weight, 0.0f, 0.02f)
        case _ => // Other types of modules are not processed
      }
      // initialize the children modules
      module.children().foreach(initModule)
    }
    // use the default initialization
    initModule(this)
  }

  def apply(
      idx: Tensor[ParamType],
      targets: Option[Tensor[ParamType]] = None
  ): (Tensor[ParamType], Option[Tensor[ParamType]]) = forward(idx, targets)

  def forward(
      idx: Tensor[ParamType],
      targets: Option[Tensor[ParamType]] = None
  ): (Tensor[ParamType], Option[Tensor[ParamType]]) = {
    val (batch, seqLen) = (idx.shape(0).toInt, idx.shape(1).toInt)
    // get the token embedding and position embedding
    val tokenEmb = tokenEmbeddingTable(idx)
    val posIndices = torch.arange(end = seqLen, device = idx.device)
    val posEmb = positionEmbeddingTable(posIndices)
    // output embedding
    var x = (tokenEmb + posEmb.unsqueeze(0).expand(batch, -1, -1)).to(idx.dtype)

    x = blocks(x)
    // final layer normalization
    x = lnFinal(x)
    // compute the output logits
    val logits = lmHead(x)
    println("Debug: compute loss (if targets are provided)")
    val loss = targets.map { target =>
      val (batch, seqLen, vocabSize) =
        (logits.shape(0).toInt, logits.shape(1).toInt, logits.shape(2).toInt)
      val reshapedLogits = logits.reshape(batch * seqLen, vocabSize)
      val reshapedTargets = target.reshape(batch * seqLen)
      F.cross_entropy(reshapedLogits.to(float32), reshapedTargets)
    }

    (logits, loss.map(_.to(idx.dtype)))
  }

  // generate text
  def generate(idx: Tensor[ParamType], maxNewTokens: Int): Tensor[ParamType] = {
    var result = idx.clone()

    for (_ <- 0 until maxNewTokens) {
      // limit the sequence length
      val neBlockSize = -config.block_size
      val idxCond = if (result.shape(1).toInt > config.block_size) {
        result(::, neBlockSize.::(result.shape(1).toInt))
      } else {
        result
      }
      // get the logits
      val (logits, _) = forward(idxCond)
      // focus on the last time step
      val lastLogits = logits(::, -1, ::)
      // apply softmax to get probabilities
      val probs = F.softmax(lastLogits, -1)
      // sample the next token
      val idxNext = torch.multinomial(probs, 1)
      // append the next token to the sequence
      result = torch.cat(Seq(result, idxNext.to(result.dtype)), 1)
    }

    result
  }
}
