import os.path

import onnx

from clashvision.core.path import get_models_path

if __name__ == '__main__':
    # Path to your input and output models
    input_path = os.path.join(get_models_path(), "v1", "best_ir9.onnx")
    output_path = os.path.join(get_models_path(), "v1", "best_ir9_opset19.onnx")

    # Load the model
    model = onnx.load(input_path)

    # Convert to opset 19 (compatible with your Rust ONNX Runtime)
    # This function updates the opset imports in the model
    onnx.checker.check_model(model)

    # Update opset to 19
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = 19

    # Save the updated model
    onnx.save(model, output_path)

    print(f"Converted model saved to: {output_path}")
