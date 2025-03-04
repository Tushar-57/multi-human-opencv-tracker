import cv2
import os
import sys

def test_model_load():
    # Define model paths
    model_path = os.path.join('models', 'mobilenet_ssd.caffemodel')
    config_path = os.path.join('models', 'deploy.prototxt')

    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    try:
        # Load the network
        print("Loading network...")
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        
        # Get layer names
        layer_names = net.getLayerNames()
        print("\nNetwork loaded successfully!")
        print(f"Number of layers: {len(layer_names)}")
        
        # Print information about each layer
        print("\nLayer Information:")
        for i, layer_name in enumerate(layer_names):
            try:
                layer = net.getLayer(i)
                print(f"Layer {i}: Name={layer_name}, Type={layer.type}")
            except:
                print(f"Layer {i}: Name={layer_name}, Type=Unknown")
        
        # Get network input details
        input_blob = net.getUnconnectedOutLayersNames()
        print(f"\nOutput layers: {input_blob}")
        
    except Exception as e:
        print(f"Error loading or analyzing the network: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_model_load()

