from yolo_processor import load_yolo_model, process_image
from risk_register import generate_risk_register

def main():
    # Extract entities from the image
    vision_model = load_yolo_model('./model_weights/yolo-11-detector-weights.pt')
    image_path = './examples/test.jpg'
    confidence_threshold = 0.4
    entities = process_image(vision_model, image_path, confidence_threshold)

    # Generate Risk Register
    llm = "openai"
    useRAG = True
    print("generating risk register")
    risk_register = generate_risk_register(entities=entities, model=llm, rag=useRAG)

    # Output the risk register
    print(risk_register)

if __name__ == "__main__":
    main() 