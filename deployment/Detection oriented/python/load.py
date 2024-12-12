from openvino.runtime import Core
import cv2
import numpy as np

# Initialize the OpenVINO runtime Core
ie = Core()

# Load the model
model = ie.read_model(model="../model/model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Load an example image
image = cv2.imread("./test.jpg")

# Resize the image to match the model's input size (1024x1024)
image_resized = cv2.resize(image, (1024, 1024))

# Convert the image to the format required by the model
# Normalize to [0, 1] if required
image_normalized = image_resized.astype(np.float32) / 255.0

# Change image format from NHWC (H, W, C) to NCHW (C, H, W)
input_data = np.transpose(image_normalized, (2, 0, 1))  # Now (3, 1024, 1024)

# Add batch dimension to make the shape [1, 3, 1024, 1024]
input_data = np.expand_dims(input_data, axis=0)

# Run inference
results = compiled_model([input_data])
print(results)
