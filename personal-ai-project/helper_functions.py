from pinecone import Pinecone
import google.generativeai as genai
import textwrap

from dotenv import load_dotenv
import os
load_dotenv()

## Initializing connection to Pinecone index
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("general-handbook-2")

## Initializing Google API to generate embeddings
google_api_key=os.environ.get("GOOGLE_API_KEY")

## Creating text generator
text_generator = genai.GenerativeModel('gemini-1.5-flash')

genai.configure(api_key=google_api_key)

def query_pinecone_with_id(query_text, top_k=5, context_window=3):
    """
    Query Pinecone with the given query text and return a context window based on the reference vector's ID.
    
    Args:
        query_text (str): The query text to search.
        top_k (int): Number of top results to return.
        context_window (int): Number of surrounding vectors to include before and after the reference vector.
        
    Returns:
        list: A list of formatted strings with the reference metadata at the top and the surrounding content.
    """
    
    # Step 1: Embed the query text using Gemini API
    query_vector = genai.embed_content(
        model="models/embedding-001",
        content=query_text,
        task_type="retrieval_document",
        title="Embedding of single string"
    )

    # Step 2: Query Pinecone for the top result (reference vector)
    results = index.query(
        namespace="general-handbook-vectors-2",
        vector=query_vector["embedding"],
        top_k=top_k,  # Retrieve just the top result
        include_values=False,
        include_metadata=True
    )

    # Step 3: Find the top result (reference vector) and its ID
    top_result = results["matches"][0]
    reference_id = int(top_result["id"])  # Convert the ID to an integer
    
    # Step 4: Retrieve the exact preceding and following vectors using the reference ID
    surrounding_ids = list(range(reference_id - context_window, reference_id + context_window + 1))

    # Step 5: Query Pinecone for each surrounding vector by ID
    context_matches = []
    for vector_id in surrounding_ids:
        try:
            result = index.query(
                namespace="general-handbook-vectors-2",
                id=str(vector_id),  # Use the vector ID as a string
                top_k=1,  # Get just the vector with this ID
                include_values=False,
                include_metadata=True
            )
            context_matches.append(result["matches"][0])  # Add the result to the context matches list
        except:
            # If a vector with the ID doesn't exist, skip it
            continue
    
    # "metadata": {"chapter": "example", "chapter_url": "example", "section": "example", "title": "example", "content": "example"}
    # Step 6: Format the metadata for the reference vector
    reference_metadata = textwrap.dedent(f"""
    Reference Metadata From Top Result:
    Chapter: {top_result['metadata']['chapter']}
    Section Title: {top_result['metadata']['title']}
    Section Number: {top_result['metadata']['section']}
    Url: {top_result['metadata']['chapter_url']}
    --------------------------------------------------""")

    # Step 7: Format the surrounding context (excluding metadata, just content)
    preceding_context = []
    following_context = []
    
    for match in context_matches:
        if match["id"] == str(reference_id):
            continue  # Skip the reference vector itself
        elif int(match["id"]) < reference_id:
            preceding_context.append(textwrap.dedent(f"Content: {match['metadata']['content']} (Chapter: {match['metadata']['chapter']}, Section: {match['metadata']['section']} {match['metadata']['title']})"))
        else:
            following_context.append(textwrap.dedent(f"Content: {match['metadata']['content']} (Chapter: {match['metadata']['chapter']}, Section: {match['metadata']['section']} {match['metadata']['title']})"))

    # Step 8: Combine the preceding context, reference metadata, and following context
    final_output = [reference_metadata] + preceding_context + [f"Content: {top_result['metadata']['content']} (Chapter: {match['metadata']['chapter']}, Section: {match['metadata']['section']} {match['metadata']['title']})"] + following_context
    print("Top result: ", top_result["metadata"]["content"])

    match_string = "\n".join(final_output)

    return match_string

def format_chat_prompt(message, chat_history):
    """
    Format the chat history and current user message into a string prompt.

    The resulting string is a multi-line string that shows the entire chat history
    with the current user message at the end. The format is as follows:
    User: <user message>
    Assistant: <assistant response>
    ... (for each turn in the chat history)
    User: <current user message>
    Assistant: (to be filled in by the chatbot)

    This string is suitable for input into a chatbot language model.

    Parameters
    ----------
    message : str
        The current user message
    chat_history : list of tuples
        The chat history, where each tuple contains the user's input message and the bot's response

    Returns
    -------
    prompt : str
        The formatted prompt string
    """
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
        
        #information = query_pinecone(message)        
    """
    Generate a response to the given message, given the chat history.

    This function is called by the Gradio interface to generate a response to the user's input.
    It uses the query_pinecone_with_id function to query the vector database and get results,
    and then formats those results into a prompt string for the text generator.
    The text generator is then used to generate a response to the user's input.
    The response is then added to the chat history and returned to the user.

    Parameters
    ----------
    message : str
        The user's input message
    chat_history : list of tuples
        The chat history, where each tuple contains the user's input message and the bot's response

    Returns
    -------
    bot_message : str
        The bot's response to the user's input
    chat_history : list of tuples
        The updated chat history
    """
    information = query_pinecone_with_id(message, top_k=7, context_window=5)
    print(information)

    full_prompt = textwrap.dedent(f"""
    Your role:
    You are a chatbot that answers questions about the General Handbook of the Church of Jesus Christ of Latter Day Saints.

    Intructions:
    - Below is a list of information received from a vector database where you might find information to answer the user's question about the General Handbook.
    - The information contains a chapter, section title and number, and a url from the top result. It also contains the content of the top result and neighbor results, aka context window.
    - You will answer questions using the content. Always include references for the user if available (Chapter, section title, section number, and url), including a link using the URL of the top result.
    - If the user wants to speak in another language, use their language in your answer.

    Information:
    START OF INFORMATION

    {information}

    END OF INFORMATION

    Further instructions: 
    - If the user seems to be asking about something not related to the General Handbook, invite them to ask about it. 
    - If instead of a questions they seem to be thanking you for previous answers, show that you welcome their gratitude.

    User Input or Question:
    {message}
    """)


    formatted_prompt = format_chat_prompt(full_prompt, chat_history)
    bot_message = text_generator.generate_content(formatted_prompt).text
    chat_history.append((message, bot_message))
    return "", chat_history