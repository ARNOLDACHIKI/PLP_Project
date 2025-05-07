# quantization.py
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def convert_to_onnx(model, model_path='models/battery_predictor.onnx'):
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model converted to ONNX and saved at {model_path}")
    return model_path

def quantize_model(onnx_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantized_model_path = onnx_model_path.replace(".onnx", "_quantized.onnx")
    quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)
    print(f"Model quantized and saved at {quantized_model_path}")
    return quantized_model_path
