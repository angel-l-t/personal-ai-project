import gradio as gr
from helper_functions import respond

with gr.Blocks(theme=gr.themes.Soft(), css=".main {max-width: 800px; margin: auto}") as demo:
    # Add a title and description at the top
    gr.Markdown("""
    # General Handbook Chatbot
    Ask questions about the General Handbook of the Church of Jesus Christ of Latter-Day Saints.
    
    Type your question below and get relevant answers based on the content of the handbook.
    """)
    
    chatbot = gr.Chatbot(height=480)  # Fixed height for the chatbot box
    msg = gr.Textbox(label="Question")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")
    
    # Add clickable example questions at the bottom
    examples = gr.Examples(
        examples=["What are the responsibilities of the Elder's Quorum President?", 
                  "What are the questions for the temple recommend interview?", 
                  "What does the church say about immigration?",
                  "Ward and stake callings chart."],
        inputs=msg
    )

    # Button click and submit actions
    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

gr.close_all()
demo.launch(share=True)