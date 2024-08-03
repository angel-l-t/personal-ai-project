from pinecone import Pinecone
import google.generativeai as genai

from dotenv import load_dotenv
import os
load_dotenv()

## Initializing connection to Pinecone index
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("general-handbook")

## Initializing Google API to generate embeddings
google_api_key=os.environ.get("GOOGLE_API_KEY")

## Creating text generator
text_generator = genai.GenerativeModel('gemini-1.5-flash')

genai.configure(api_key=google_api_key)

def query_pinecone(query_text, top_k=5):
    """
    This function queries the Pinecone index for the given query text and returns the top-k results as a list of formatted strings.
    args:
        query_text: string
        top_k: integer
    """
    
    query_vector = genai.embed_content(
        model = "models/embedding-001",
        content = query_text,
        task_type = "retrieval_document",
        title = "Embedding of single string"
    )

    results = index.query(
        namespace = "general-handbook-vectors",
        vector = query_vector["embedding"],
        top_k = top_k,
        include_values = False,
        include_metadata = True,
        # filter = {"genre": {"$eq": "action"}}
    )

    match_list = []

    for matched in results["matches"]:
        match_string = f"""
        Chapter: {matched['metadata']['chapter']}
        Header: {matched['metadata']['title']}
        Section: {matched['metadata']['section']}
        Url: {matched['metadata']['url']}
        --------------------------------------------------
        Content: {matched['metadata']['content']}
        """

        match_list.append(match_string)
        match_string = "\n".join(match_list)

    return match_string

def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        
        information = query_pinecone(message)
        information = "\n".join(information)

        full_prompt = f"""
        Your role:
        You are a chatbot that answers questions about the General Handbook of the Church of Jesus Christ of Latter Day Saints.

        Intructions:
        Below is a list of information received from a vector database where you might find information to answer the user's question about the General Handbook.
        Each chunk of information contains a title, url, and content. You will mainly answer questions from the content, but feel free to share that extra information as reference.

        Information:
        START OF INFORMATION

        {information}

        END OF INFORMATION

        (Further instructions: If the user seems to be asking about something not related to the General Handbook, invite them to ask about it. If instead of a questions they seem to be thanking you for previous answers, show that you welcome their gratitude)
        User Input:
        {message}
        """


        formatted_prompt = format_chat_prompt(full_prompt, chat_history)
        bot_message = text_generator.generate_content(formatted_prompt).text
        chat_history.append((message, bot_message))
        return "", chat_history