import onnx
import onnxscript
from onnxscript.rewriter import pattern
from onnxscript import ir

gelu_model = onnx.load("gelu_model.textproto")

def erf_gelu_pattern(op, x):
    return 0.5 * (x * (op.Erf(x / math.sqrt(2)) + 1.0))

def gelu(op, x: ir.Value):
    return op.Gelu(x, domain="com.microsoft")

