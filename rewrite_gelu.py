import math

import onnx
import onnxscript
from onnxscript import ir
from onnxscript.rewriter import pattern

gelu_model = ir.from_proto(onnx.load("gelu_model.textproto"))


def erf_gelu_pattern(op, x):
    return 0.5 * (x * (op.Erf(x / math.sqrt(2)) + 1.0))


def gelu(op, x: ir.Value):
    return op.Gelu(x, domain="com.microsoft")


def apply_rewrite_with_commute(model: ir.Model) -> ir.Model:
    rule = pattern.RewriteRule(
        erf_gelu_pattern,  # Target Pattern
        gelu,  # Replacement
    )
    # Create a Rewrite Rule Set with commute=True
    rewrite_rule_set = pattern.RewriteRuleSet([rule], commute=True)
    # Apply rewrites
    model_with_rewrite_applied = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=rewrite_rule_set,
    )
    return model_with_rewrite_applied


# Apply rewrite
# Apply rewrite
fused_relu_model = apply_rewrite_with_commute(gelu_model)
onnx.save(fused_relu_model, "fused_relu_model.onnx")
print(fused_relu_model)

# Applied 1 of general pattern rewrite rules.
# <
#     ir_version=9,
#     opset_imports={'': 18, 'pkg.onnxscript.torch_lib': 1, 'com.microsoft': 1},
#     producer_name='torch',
#     producer_version='2.3.1',
#     domain=None,
#     model_version=None,
# >
# graph(
#     name=main_graph,
#     inputs=(
#         %"arg0_1"<FLOAT,[1,1]>
#     ),
#     outputs=(
#         %"val_gelu"<FLOAT,[1,1]>
#     ),
# ) {
#     0 |  # node_Gelu_0
#          %"val_gelu"<FLOAT,[1,1]> ⬅️ com.microsoft::Gelu(%"arg0_1")
#     return %"val_gelu"<FLOAT,[1,1]>
