package torch
package nn
package modules
package graph

// Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)
case class GraphData[ParamDType <: DType](
    x: Tensor[ParamDType],
    edge_index: Tensor[Int64],
    edge_attr: Tensor[ParamDType],
    y: Tensor[ParamDType]
)
