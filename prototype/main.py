from yolo_processor import load_yolo_model, process_image
from risk_register import generate_risk_register

def main():
    # Fake Entities for testing
    print("creating fake entities")
    test_entities_1 = [
        {"id": 1, "class": "weed", "bbox": [0.3, 0.5, 0.1, 0.1]},
        {"id": 2, "class": "water", "bbox": [0.35, 0.55, 0.08, 0.09]},
        {"id": 3, "class": "native_plant", "bbox": [0.6, 0.5, 0.12, 0.15]},
        {"id": 4, "class": "track", "bbox": [0.5, 0.8, 0.2, 0.05]}
    ]
    test_entities_2 = [
        {"id": 1, "class": "native_plant", "bbox": [0.3, 0.5, 0.1, 0.1]},
        {"id": 2, "class": "water", "bbox": [0.35, 0.55, 0.08, 0.09]},
        {"id": 3, "class": "native_plant", "bbox": [0.6, 0.5, 0.12, 0.15]},
        {"id": 4, "class": "track", "bbox": [0.5, 0.8, 0.2, 0.05]}
    ]
    # Extract entities from the image
    #vision_model = load_yolo_model('./model_weights/yolo-11-detector-weights.pt')
    #image_path = 'path/to/your/image.jpg'
    #entities = process_image(vision_model, image_path)

    # Generate Risk Register
    llm = "openai"
    print("generating risk register")
    risk_register = generate_risk_register(test_entities_2, llm)

    # Output the risk register
    print(risk_register)

if __name__ == "__main__":
    main() 