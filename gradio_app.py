# gradio_app.py

import gradio as gr
from rizzbot_agentic import Rizzbot

# Initialize the Rizzbot instance
rizzbot = Rizzbot()

# Function to interface with Gradio
def respond_to_question(user_input):
    return rizzbot.answer_question(user_input)

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Base(), css="body {background-color: #000000; color: #FFFFFF;}") as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: white;'>Welcome to Rizzbot: the chatbot to help you autists become better communicators!</h1>"
    )

    with gr.Column():
        gr.Markdown("<p style='color: white;'>Please type your Rizz-related question here</p>")
        user_input = gr.Textbox(placeholder="e.g., What are some good practices when giving a presentation?", label=None)
        submit_btn = gr.Button("Submit")
        output = gr.Textbox(label="Rizzbot's Response", lines=5, interactive=False)

    submit_btn.click(fn=respond_to_question, inputs=user_input, outputs=output)

# Launch the app
if __name__ == "__main__":
    demo.launch()

