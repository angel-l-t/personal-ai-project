import gradio as gr
from helper_functions import respond

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=480) #just to fit the notebook
    msg = gr.Textbox(label="Question")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()

if __name__ == "__main__":
    demo.launch(share=True)