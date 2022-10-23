import cv2

# Set up YOLO Layers
def get_output_layers(net):
    # Get the names of the layers in the network
    layer_names = net.getLayerNames()
    
    # Get the names of the output layers i.e. the unconnected layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Read Image
def read_image(image_path):
    image = cv2.imread(image_path)
    return image
