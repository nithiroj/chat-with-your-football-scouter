## Chat with Your Football Scouter

Seeking insights on your target football players? Chat with Your Scouter! Get in-depth scouting reports, latest info, market value, strengths, weaknesses, and comparisons. Everything you need to know for your target player.

## Description

We use 2022-2023 Football Player Stats from Kaggle. The data encompasses nearly 2500 players across Premier League, Ligue 1, Bundesliga, Serie A, and La Liga. Covering 125 metrics, ranging from basic player information such as name, age, and nation, to performance statistics like goals and pass completion rates, our dataset is extensive and diverse. To harness and organize this wealth of information, we leverage Cohere Embedding and Weaviate Cloud Service (WCS), employing vector transformation, storage, and indexing.

The focal points of our application are the Chat and Compare Player features, each powered by advanced language models, including Cohere and ChatCohere. Both functionalities employ Retrieval-augmented Generation (RAG) techniques, albeit with distinct details and components.

For the Chat feature, we've constructed a compressor retriever using Cohere Rag Retriever, incorporating a web-search connector and CohereRerank as a compressor. Within the ConversationBufferMemory chain, this chain processes chat history (a list of messages) and new questions, ultimately delivering a response. The algorithm in this chain comprises three key steps: first, the creation of a "standalone question" using chat history and the new question; second, passing this question to the retriever to fetch relevant documents; and finally, utilizing the retrieved documents in conjunction with either the new question or the original question and chat history to generate a comprehensive response.

Conversely, the Player Comparison feature utilizes the Weaviate Hybrid Search Retriever to extract statistical data of players by their names from WCS. Through an LLM chain, we then summarize this data and generate a comprehensive report based on the retrieved documents. Our approach ensures a robust and dynamic platform for users seeking nuanced insights into player performances across top football leagues.
