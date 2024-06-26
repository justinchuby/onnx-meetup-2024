
Construction

```py
from onnxscript import ir

v0 = ir.Input(name="v0")
v1 = ir.Input(name="v1")
node = ir.Node(
    "", "Add", inputs=(v0, v1), num_outputs=1, name="node_add"
)
graph = ir.Graph(
    (v0, v1),
    node.outputs,
    nodes=(node,),
    opset_imports={"": 1},
)
```
