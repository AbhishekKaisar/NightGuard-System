import os
import torch
import yaml
from modules.enhancement.models.ensemble import LowLightEnsemble, load_base_weights
from modules.enhancement.models.base_models import ZeroDCE, KinD, RetinexNet, Restormer
from modules.enhancement.utils.helpers import _resolve_path, _resolve_weight_paths

def export_onnx():
    device = torch.device('cpu')

    print("Loading models...")
    dce = ZeroDCE()
    kind = KinD()
    retinex = RetinexNet()
    restormer = Restormer(LayerNorm_type='BiasFree')

    ensemble = LowLightEnsemble(dce, kind, retinex, restormer).to(device)

    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "enhancement")
    config_path = os.path.join(project_root, "configs", "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _resolve_weight_paths(cfg, project_root)
        load_base_weights(ensemble, cfg, device)
        
        fusion_weight_path = os.path.join(
            _resolve_path(project_root, cfg['weights']['save_dir']),
            cfg['weights']['save_name']
        )
        
        if os.path.exists(fusion_weight_path):
            ensemble.fusion_unet.load_state_dict(torch.load(fusion_weight_path, map_location=device))
        else:
            print(f"Warning: Could not find fusion weights at {fusion_weight_path}")
    else:
        print(f"Warning: Could not find config at {config_path}")

    ensemble.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # Ensure output directory exists
    os.makedirs('weights', exist_ok=True)
    onnx_path = 'onnx_weights/ensemble_fp32.onnx'
    onnx_fp16_path = 'onnx_weights/ensemble_fp16.onnx'

    print(f"Exporting ONNX model to {onnx_path}...")
    torch.onnx.export(
        ensemble,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    print("ONNX export completed.")

    print(f"Converting to FP16: {onnx_fp16_path}...")
    import onnx
    from onnxconverter_common import float16
    
    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, onnx_fp16_path)
    print("FP16 conversion completed.")

if __name__ == '__main__':
    export_onnx()
