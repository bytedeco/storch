package torch.distribute

enum WorkResult(val value: Byte):
  case SUCCESS extends WorkResult(0)
  case TIMEOUT extends WorkResult(1)
  case COMM_ERROR extends WorkResult(2)
  case UNKNOWN extends WorkResult(100)

  /** 查找并返回与当前value匹配的枚举单例实例，逻辑与Java版`intern()`一致
    */
  def intern(): WorkResult =
    WorkResult.values.find(_.value == this.value).getOrElse(this)

  /** 重写toString，返回interned实例的名称
    */
//  override def toString(): String = intern().name
