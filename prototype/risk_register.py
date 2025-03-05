from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

# Define a Pydantic model for individual risks
class Risk(BaseModel):
    risk_id: int = Field(description="Unique identifier for the risk")
    risk_type: str = Field(description="Type of risk, e.g., Growth, Competition, Propagation")
    associated_entities: str = Field(description="List of interacting entities by class and id, e.g., 'Entity 1 (Weed) & Entity 2 (Water)'")
    assessment: str = Field(description="Description of the possible interactions causing the risks and the likelihood of the risk occurring based on the spatial relationships between the entities")
    justification: str = Field(description="Reference to a supporting article/rule provided that was used to justify the risk assessment")

# Update the RiskRegister model to include a list of Risk models
class RiskRegister(BaseModel):
    risks: list[Risk] = Field(description="List of identified risks")
    
def load_graph_docs():
    doc_list = ["prototype/docs/competition.md", "prototype/docs/parkinsonia.md", "prototype/docs/tracks.md", "prototype/docs/water.md"]
    # Load the RAG documents
    docs = []
    for doc_name in doc_list:
        loader = UnstructuredMarkdownLoader(doc_name)
        docs.extend(loader.load())  # Use extend instead of append to flatten the list
    
    # Split the documents into chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph_transformer = LLMGraphTransformer(llm=llm)
    graph_docs = graph_transformer.convert_to_graph_documents(chunks)
    print(graph_docs)
    return graph_docs

def initialise_graph(documents):
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        enhanced_schema=True,
    )
    graph.add_graph_documents(documents, include_source=True, baseEntityLabel=True)
    return graph

def get_graph():
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        enhanced_schema=True,
    )
    return graph

def initialise_model(llm: str):
    print(f"initialising model: {llm}")
    if llm == "mistral":
        chat_model = ChatMistralAI(model="mistral-large-latest", temperature=0)
    elif llm == "openai":
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"Unsupported language model: {llm}")
    print(f"model initialised: {chat_model}")
    return chat_model
    
def initialise_prompt(entities, documents=None):
    # System prompt
    sys_prompt = """You are a helpful assistant that takes the output of an object detector (consisting of entities and their positions) and generates a risk register. We are interested in promoting native rehabilitation of the land. Only identify negative risks caused by weeds.
                    Analyse the spatial relationships between the entities in the image using their relative positions. Identify potential risks based on interactions if they exist (e.g., proximity, overlap, or contextual patterns) between pairs of entities. Return the risks in a JSON-like format as a dictionary with a key "risks" containing a list of dictionaries, each representing a risk with the following keys:
                    
                    - "risk_id": Unique identifier for the risk.
                    - "risk_type": Type of risk, e.g., Growth, Competition, Propagation.
                    - "associated_entities": A single string listing the interacting entities by class and id, e.g., "Entity 1 (Weed) & Entity 2 (Water)".
                    - "assessment": Description of the possible interactions causing the risks and the likelihood of the risk occurring based on the spatial relationships between the entities.
                    - "justification": Reference to a supporting article/rule provided that was used to justify the risk assessment. Relevent documents will be provided prior to the entities if necessary. 

                    **Example Output:**
                    {{
                        "risks": [
                            {{
                                "risk_id": 1,
                                "risk_type": "Growth",
                                "associated_entities": "Entity 1 (Weed) & Entity 2 (Water)",
                                "assessment": "High chance of weed growth",
                                "justification": "Article 1"
                            }},
                            {{
                                "risk_id": 2,
                                "risk_type": "Competition",
                                "associated_entities": "Entity 1 (Weed) & Entity 3 (Native Plant)",
                                "assessment": "Weed may outcompete native species",
                                "justification": "Article 2"
                            }}
                        ]
                    }}
                    
                    In the case of no risks, simply return an empty dictionary with "risks" as an empty list: {{"risks": []}}. The user will now provide the list of entities and their bounding box positions.
                """
    # Initialise the prompt template
    print(f"initialising prompt")
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "Documents: {documents}"),
        ("user", "Entities: {entities}"),
    ])
    return prompt

def retrieve_documents(entities, graph):
    chain = GraphCypherQAChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-4o-mini"),
        graph=graph,
        verbose=True,
        validate_cypher=True,
        allow_dangerous_requests=True,
    )
    entity_types = []
    for entity in entities:
        entity_types.append(entity["class"])
        
    query = "How do these entitites relate to each other: {entity_types}?"
    result = chain.invoke({"query": query}) 
    
    return result

def initialise_parser():
    parser = PydanticOutputParser(pydantic_object=RiskRegister)
    return parser

def generate_risk_register(entities, llm):
    # initialise the model
    chat_model = initialise_model(llm)
    
    # Initialise the prompt 
    prompt = initialise_prompt(entities)

    # initialise the parser
    parser = initialise_parser()

    # create the pipeline
    print(f"creating pipeline")
    pipeline = prompt | chat_model | parser
    
    print(f"invoking pipeline")
    output = pipeline.invoke({"entities": entities})
    return output

mock_entities = [
    {
        "id": 0,
        "class": "parkinsonia",
        "bbox": [150, 200, 50, 120],  # x, y, width, height
        "interactions": [
            "Distance between entity 0 (parkinsonia) and entity 1 (water) is 3.75 metres",
            "Distance between entity 0 (parkinsonia) and entity 2 (native_plant) is 2.5 metres"
        ]
    },
    {
        "id": 1,
        "class": "water",
        "bbox": [300, 250, 200, 150],
        "interactions": [
            "Distance between entity 1 (water) and entity 0 (parkinsonia) is 3.75 metres",
            "Distance between entity 1 (water) and entity 2 (native_plant) is 5.0 metres"
        ]
    },
    {
        "id": 2,
        "class": "native_plant",
        "bbox": [200, 300, 40, 30],
        "interactions": [
            "Distance between entity 2 (native_plant) and entity 0 (parkinsonia) is 2.5 metres",
            "Distance between entity 2 (native_plant) and entity 1 (water) is 5.0 metres"
        ]
    }
]

test = retrieve_documents(mock_entities, get_graph())
print(test)