from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

# Create a Pydantic model for the risk register
class RiskRegister(BaseModel):
        risk_register: str = Field(description="The risk register table for the entities")

def initialise_model(llm: str):
    print(f"initialising model: {llm}")
    if llm == "mistral":
        chat_model = ChatMistralAI(model="mistral-large-latest")
    elif llm == "openai":
        chat_model = ChatOpenAI(model="gpt-4o-mini")
    else:
        raise ValueError(f"Unsupported language model: {llm}")
    print(f"model initialised: {chat_model}")
    return chat_model
    
def initialise_prompt(entities):
    # System prompt
    sys_prompt = """You are a helpful assistant that takes the output of an object detector (consisting of entities and their positions) and generates a risk register. We are interested in promoting native rehabilitation of the land. Only identify negative risks caused by weeds 
                    Analyse the spatial relationships between the entities in the image using their bounding box positions. Identify potential risks based on interactions if they exist (e.g., proximity, overlap, or contextual patterns) between pairs of entities. In the case of risks generate a risk register with the following columns:  
                    1. **Risk ID** (Unique numeric identifier).  
                    2. **Risk Type** (e.g., Growth, Competition, Propagation).  
                    3. **Associated Entities** (List the interacting entities by class, e.g., "Weed & Water").  
                    4. **Assessment** (Brief risk description, e.g., "High chance of weed growth due to nearby water source").  
                    5. **Justification** (Reference to a supporting article/rule, e.g., "Article 1").  

                    **Examples:**  
                    | Risk ID | Risk Type   | Associated Entities      | Assessment                          | Justification |
                    |---------|-------------|--------------------------|-------------------------------------|---------------|
                    | 1       | Growth      | Weed & Water             | High chance of weed growth          | Article 1     |
                    | 2       | Competition | Weed & Native Plant      | Weed may outcompete native species  | Article 2     |    
                    | 3       | Propagation | Weed & tracks            | Weed may spread to other areas      | Article 3     |
                    
                    In the case of no risks, simply state "No risks identified". The user will now provide the list of entities and their bounding box positions.
                """
    # Initialise the prompt template
    print(f"initialising prompt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "Entities: {entities}"),
    ])
    return prompt

def initialise_parser():
    parser = PydanticOutputParser(pydantic_object=RiskRegister)
    return parser

def generate_risk_register(entities, llm):
    # initialise the model
    chat_model = initialise_model(llm)
    
    # Initialise the prompt 
    prompt = initialise_prompt(entities)

    # initialise the parser
    # parser = initialise_parser()

    # create the pipeline
    print(f"creating pipeline")
    pipeline = prompt | chat_model
    
    print(f"invoking pipeline")
    output = pipeline.invoke({"entities": entities})
    return output.content 



