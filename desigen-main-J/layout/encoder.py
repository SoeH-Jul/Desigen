import jittor as jt
from jittor import models as jmodels

class EncoderWrapper(jt.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward_features(self, x):
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        return self.model(x)

    def execute(self, x):
        return self.forward_features(x)

def _safe_create(model_fn, **kwargs):
    try:
        return model_fn(**kwargs)
    except TypeError:
        kwargs.pop("num_classes", None)
        return model_fn(**kwargs)

def _get_model_by_name(name, pretrain):
    if name == 'vit': # ([1, 50, 768])
        if hasattr(jmodels, "vit_base_patch32_224"):
            return _safe_create(jmodels.vit_base_patch32_224, pretrained=pretrain, num_classes=0)
        if hasattr(jmodels, "vit_base_patch16_224"):
            return _safe_create(jmodels.vit_base_patch16_224, pretrained=pretrain, num_classes=0)
        raise NotImplementedError("Jittor vit model not available.")
    if name == 'swin': # ([1, 49, 1024])
        if hasattr(jmodels, "swin_base_patch4_window7_224"):
            return _safe_create(jmodels.swin_base_patch4_window7_224, pretrained=pretrain, num_classes=0)
        if hasattr(jmodels, "swin_tiny_patch4_window7_224"):
            return _safe_create(jmodels.swin_tiny_patch4_window7_224, pretrained=pretrain, num_classes=0)
        raise NotImplementedError("Jittor swin model not available.")
    if name == 'resnet': # ([1, 2048, 7, 7])
        if hasattr(jmodels, "resnet50"):
            return _safe_create(jmodels.resnet50, pretrained=pretrain, num_classes=0)
        raise NotImplementedError("Jittor resnet model not available.")
    if name == 'clip':
        raise NotImplementedError("Jittor clip encoder not available.")
    raise NotImplementedError

def get_encoder(name='vit', grad=False, pretrain=True):
    print(f'encoder status: name-{name}, grad-{grad}, pretrain-{pretrain}')
    model = _get_model_by_name(name, pretrain)
    
    # disable gradient    
    if not grad:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    
    return EncoderWrapper(model)

if __name__ == '__main__':
    # NOTE: no pretrained
    model = get_encoder(name='clip')
    input = jt.randn(1, 3, 224, 224)
    out = model.forward_features(input)
    print(out.shape) 
