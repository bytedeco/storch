package torch.utils.data.dataset.custom

import org.bytedeco.pytorch.ExampleVector
import torch.utils.data.Dataset
import torch.*

class NetflixPrizeDataset[
    Input <: BFloat16 | FloatNN: Default,
    Target <: BFloat16 | FloatNN: Default
] extends Dataset[Input, Target] {

  // https://ai.gitcode.com/datasets?utm_source=highlight_word_gitcode&word=data
  // https://tianchi.aliyun.com/dataset/146311/
  val DATA_URL =
    "https://tianchi-jupyter-sh.oss-cn-shanghai.aliyuncs.com/file/opensearch/documents/146311/Netflix_Dataset.zip?Expires=1758998949&OSSAccessKeyId=STS.NYUBo81jYSm1ZAU9xc4UP44nq&Signature=FByO0Lr9QQWads9omGC3ijVWj3w%3D&response-content-disposition=attachment%3B%20&security-token=CAIS0wJ1q6Ft5B2yfSjIr5rgCdWM3LV45K%2FaWGfk3Xg2ONp82%2FHFkzz2IHhMeXdvCeAfsvs1lG9W7vYYlrp6SJtIXleCZtF94oxN9h2gb4fb41FUaTX008%2FLI3OaLjKm9u2wCryLYbGwU%2FOpbE%2B%2B5U0X6LDmdDKkckW4OJmS8%2FBOZcgWWQ%2FKBlgvRq0hRG1YpdQdKGHaONu0LxfumRCwNkdzvRdmgm4NgsbWgO%2Fks0aH1Q2rlbdM%2F9WvfMX0MvMBZskvD42Hu8VtbbfE3SJq7BxHybx7lqQs%2B02c5onAXQYBs0zabLCErYM3fFVjGKE5H6Nft73wheU9sevOjZ%2F6jg5XOu1FguwGsiZLaaEuccPe1bZRHd6TUxylWUBd7h5X1wht5%2F5PgYmu1w69ztNUSEVKZHizAdSM0zPpeWyIIIvei5pV6vcbgiuuk5PkSFbnOAukaHpwUvcagAEbvVizhzR8UO5aFkyivTdBCtW28WitCBmTtuLIasBh3akLIxRahpS%2F%2FTrcq6nD%2FUfRKcIwqeBzuT8A0xMEpTsXbHpaMgjgX05EyRDkzztXfzF8h%2FpFudlBA3dfpQOVrKwwzFEXoTWr%2B2u%2BNsUZiDMv4LBiJvAoXqi3cxSyB%2F%2FcTyAA"

  override def length: Long = ???

  override def getItem(idx: Int): (Tensor[Input], Tensor[Target]) = ???

  override def features: Tensor[Input] = ???

  override def targets: Tensor[Target] = ???

  override def iterator: Iterator[(Tensor[Input], Tensor[Target])] = ???

  override def get_batch(request: Long*): ExampleVector = ???

}
