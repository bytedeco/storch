package torch.utils.models.llm.gpt

case class GPTConfig(
    block_size: Int = 512, // #text sequence length 这里其实应该是文本的最大长度（ max_seq_len）
    batch_size: Int = 12,
    n_layer: Int = 6,
    n_head: Int = 12,
    n_embd: Int = 768, // # n_embd 也叫 hidden_dim, hiden_size, 这里我同时设置了和 embed_dim 一样
    head_size: Int = 768, // n_head
    dropout: Float = 0.1,
//  # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
    vocab_size: Int = 50257
)

//@dataclass
//class GPTConfig:
//    block_size: int = 512   # 这里其实应该是文本的最大长度（ max_seq_len）
//    batch_size: int = 12
//    n_layer: int = 6
//    n_head: int = 12
//    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里我同时设置了和 embed_dim 一样
//    head_size: int = n_embd // n_head
//    dropout: float = 0.1
//    # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
//    vocab_size: int = 50257
//case class GptConfig(@dataclass
//case class GPTConfig(
//                      blockSize: Int = 512,  // # 这里其实应该是文本的最大长度（ max_seq_len）
//                      batchSize: Int = 12,
//                      nLayer: Int = 6,
//                      nHead: Int = 12,
//                      nEmbd: Int = 768,    //# n_embd 也叫 hidden_dim, hiden_size, 这里我同时设置了和 embed_dim 一样
//                      headSize: Int = 768, // n_head
//                      dropout: Float = 0.1,
//                      //  # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
//                      vocabSize: Int = 50257)
