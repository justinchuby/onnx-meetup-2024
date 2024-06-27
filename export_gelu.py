import torch


class GeluModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)


torch.onnx.export(GeluModel(), (torch.randn(1, 1),), "gelu_model.onnx")
