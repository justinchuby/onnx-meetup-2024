
Construction

```py
from onnxscript import ir

v0 = ir.Input(name="v0")
v1 = ir.Input(name="v1")
node_add = ir.Node("", "Add", inputs=(v0, v1))
node_cast = ir.Node(
    "", "Cast", inputs=node_add.outputs,
    attributes=ir.AttrInt64("to", ir.DataType.FLOAT16)
)
graph = ir.Graph(
    (v0, v1),
    node_cast.outputs,
    nodes=(node_add, node_cast),
    opset_imports={"": 1},
)
```
