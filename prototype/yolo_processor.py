from ultralytics import YOLO

def load_yolo_model(weights_path):
    # Load the YOLO model with the specified weights
    model = YOLO(weights_path)
    return model

def get_raw_entities(model, image_path, thresold):
    # Process the image and return the detected entities
    results = model(image_path)
    # Initialize the entities list
    entities = []
    # Iterate through the results
    for result in results:
        # Iterate through the entities
        for i, entity in enumerate(result.boxes):
            # If the confidence is greater than the threshold
            if entity.conf > thresold:
                # Get the name of the entity
                name = result.names[entity.cls.int().item()]
                # Get the bounding box of the entity
                bbox = entity.xywh.tolist()[0]
                # Append the entity to the entities list
                entities.append({"id": i, "class": name, "bbox": bbox})
    return entities

def calculate_distance(entity1, entity2):
    # pixel to metre constant
    PIXEL_TO_METER = 0.025
    # Calculate the distance between two bounding boxes
    x1, y1, _, _ = entity1["bbox"]
    x2, y2, _, _ = entity2["bbox"]
    # Calculate the distance between the two bounding boxes
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    # Convert the distance to metres
    real_distance = distance * PIXEL_TO_METER
    # Return interaction description
    return f"Distance between entity {entity1['id']} ({entity1['class']}) and entity {entity2['id']} ({entity2['class']}) is {real_distance} metres"

def process_image(model, image_path, thresold):
    # Get the raw entities
    entities = get_raw_entities(model, image_path, thresold)
    # Process the entities
    processed_entities = []
    for entity in entities:
        # Calculate the distance between the entity and the other entities
        interactions = []
        for other_entity in entities:
            if entity != other_entity:
                interaction = calculate_distance(entity, other_entity)
                interactions.append(interaction)
        # Append the entity to the processed entities list
        entity["interactions"] = interactions
        processed_entities.append(entity)
    return processed_entities

