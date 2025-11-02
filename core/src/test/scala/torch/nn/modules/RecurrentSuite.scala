package torch.nn.modules

import torch.nn

class RecurrentSuite

//rnn = nn.LSTM(10, 20, 2)
//input = torch.randn(5, 3, 10)
//h0 = torch.randn(2, 3, 20)
//c0 = torch.randn(2, 3, 20)
//output, (hn, cn) = rnn(input, (h0, c0))
class LstmSuite extends munit.FunSuite {
  test("LstmSuite output shapes") {
    val m12 = nn.LSTM(10, 20, 2)
    val input1 = torch.randn(Seq(5, 3, 10))
    val h0 = torch.randn(Seq(2, 3, 20))
    val c0 = torch.randn(Seq(2, 3, 20))
    val out = m12(input1, Some(h0), Some(c0))
    assertEquals(out._1.shape, Seq(5, 3, 20))
    println(out._1)
  }
}

//rnn = nn.GRU(10, 20, 2)
//input = torch.randn(5, 3, 10)
//h0 = torch.randn(2, 3, 20)
//output, hn = rnn(input, h0)
class GruSuite extends munit.FunSuite {
  test("GruSuite output shapes") {
    val m12 = nn.GRU(10, 20, 2)
    val input1 = torch.randn(Seq(5, 3, 10))
    val h0 = torch.randn(Seq(2, 3, 20))
    val (output, hn) = m12(input1, Some(h0))
    assertEquals(output.shape, Seq(5, 3, 20))
    println(output)
  }
}
//rnn = nn.RNN(10, 20, 2)
//input = torch.randn(5, 3, 10)
//h0 = torch.randn(2, 3, 20)
//output, hn = rnn(input, h0)
class RnnSuite extends munit.FunSuite {
  test("RnnSuite output shapes") {
    val m12 = nn.RNN(10, 20, 2)
    val input1 = torch.randn(Seq(5, 3, 10))
    val h0 = torch.randn(Seq(2, 3, 20))
    val (output, hn) = m12(input1, Some(h0))
    assertEquals(output.shape, Seq(5, 3, 20))
    println(output)
  }
}
