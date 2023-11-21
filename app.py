import os
import google.cloud.logging

from langchain.chat_models import ChatCohere
from langchain.retrievers import CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationBufferMemory
from langchain.llms import Cohere
from langchain.chains import ConversationalRetrievalChain

from langchain.schema import AIMessage, HumanMessage
import gradio as gr

import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Your Google Cloud Project ID
PROJECT_ID = os.environ.get('GCP_PROJECT', None)
# Your Google Cloud Project Region
LOCATION = os.environ.get('GCP_REGION', None)
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

if PROJECT_ID:
    client = google.cloud.logging.Client(project=PROJECT_ID)
    client.setup_logging()

    log_name = "chat-scouter-app-log"
    logger = client.logger(log_name)

chat_model = ChatCohere(model="command-nightly",
                        cohere_api_key=COHERE_API_KEY)

llm = Cohere(model="command-nightly")

### Chat ###

rag = CohereRagRetriever(llm=chat_model)

compressor = CohereRerank(user_agent="langchain")

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=rag
)

memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compressor_retriever,
    memory=memory
)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    response = chat_chain({"question": message})

    return response['answer']

### Compare Players ###


## Weaviate Client ##
WEAVIATE_URL = os.environ.get('WEAVIATE_URL')
WEAVIATE_API_KEY = os.environ.get('WEAVIATE_API_KEY')

client = weaviate.Client(
    url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY,
    }
)

## Compare chain ##

# Retriver
retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="Player",
    text_key="report",
    k=1,
    attributes=[],
    create_schema_if_missing=True,
)

# Sumarize Chain

sumarize_template = """You are an experienced professional scout.\n
Write a summarized analysis about some of the main given attributes between the following two football players:\n
1. {player1}\n
2. {player2}\n

Do emphasize that the summary based on season 2022-2023.\n

Do not ask for any further questions when finished summary.\n

"""

sumarize_prompt = PromptTemplate(
    template=sumarize_template, input_variables=["player1", "player2"])

sumarize_chain = LLMChain(prompt=sumarize_prompt, llm=llm)

# Compare Chain
compare_chain = (
    {
        'player1': lambda x: retriever.get_relevant_documents(
            x['player1'],
            where_filter={
                "path": ["player"],
                "operator": "Equal",
                "valueString": x['player1'],
            },
        ),
        'player2': lambda x: retriever.get_relevant_documents(
            x['player2'],
            where_filter={
                "path": ["player"],
                "operator": "Equal",
                "valueString": x['player2'],
            },
        )
    }
    | sumarize_chain
    | {"output_text": lambda x: x["text"]}
)


def compare_player(data):
    #  Avoid connection error
    while True:
        try:
            response = compare_chain.invoke(
                {
                    'player1': data[player1],
                    'player2': data[player2]
                }
            )
            break
        except:
            continue

    return response['output_text']


# Gradio UI
with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as demo:

    gr.Image(value=os.path.join(os.path.dirname(
        __file__), "scouter.png"), interactive=False)

    with gr.Tab("Chat"):

        gr.ChatInterface(predict)

    with gr.Tab("Compare Players"):

        with gr.Row():
            player1 = gr.Text(label="Player 1", max_lines=1,
                              placeholder="Give me a football player's name")
            player2 = gr.Text(label="Player 2", max_lines=1,
                              placeholder="Give me a football player's name")
            compare_btn = gr.Button('Compare')

        report = gr.Text(label="Report (Based on Season 2020/2023)")

        compare_btn.click(compare_player, inputs={
                          player1, player2}, outputs=report)


demo.launch(server_name="0.0.0.0", server_port=8080)  # For Cloud Run
