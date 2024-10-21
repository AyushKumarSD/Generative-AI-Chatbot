import warnings
warnings.filterwarnings('ignore')
# Import necessary libraries
import gradio as gr
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, AutoTokenizer, BartForQuestionAnswering, DistilBertTokenizerFast, DistilBertForQuestionAnswering
import gc

# Create a context store
context_store = []
selected_model = None  # To track the selected model

# Define models and tokenizers
def load_bert_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "squad-bert-trained/BERT_model"
    model = BertForQuestionAnswering.from_pretrained(model_save_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_save_path)
    model.eval().to(device)
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer, model, device

def load_bart_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartForQuestionAnswering.from_pretrained("squad-BART-trained/BART_model")
    tokenizer = AutoTokenizer.from_pretrained("squad-BART-trained/BART_model")
    model.eval().to(device)
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer, model, device

def load_distilbert_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "DISTILBERT_model"
    model = DistilBertForQuestionAnswering.from_pretrained(model_save_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_save_path)
    model.eval().to(device)
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer, model, device

# Function to generate answers with capitalized output
def generate_answer(context, question):
    try:
        max_context_size = 512
        chunk_size = max_context_size
        chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
        answers = []

        for chunk in chunks:
            inputs = tokenizer(chunk, question, return_tensors='pt', truncation=True, max_length=max_context_size).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
                answers.append(answer.capitalize())  # Capitalize the first letter of the answer

        return ' '.join(answers).strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return "❌ An error occurred while generating the answer."

# Define the Gradio interface with light theme and organized layout
def chatbot_interface():
    with gr.Blocks() as demo:
        # Custom CSS for light theme and layout
        gr.Markdown("""
            <style>
                body { background-color: #f9f9f9; }
                .chatbot-container { background-color: #ffffff; border-radius: 10px; padding: 20px; color: #333; font-family: Arial, sans-serif; }
                .gr-button { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 14px; cursor: pointer; }
                .gr-button:hover { background-color: #45a049; }
                .gr-textbox { background-color: #ffffff; color: #333; border-radius: 5px; border: 1px solid #ddd; padding: 10px; }
                .gr-chatbot { background-color: #e6e6e6; border-radius: 10px; padding: 15px; color: #333; }
                .footer { text-align: right; font-size: 12px; color: #777; font-style: italic; }
                .note { text-align: right; font-size: 10px; color: #777; font-style: italic; position: absolute; bottom: 10px; right: 10px; }
            </style>
        """)

        # Header
        gr.Markdown("<h1 style='text-align: center; color: #4CAF50;'>EDITH: Multi-Model Question Answering Platform</h1>")
        gr.Markdown("<p style='text-align: center; color: #777;'>Switch between BERT, BART, and DistilBERT models and ask questions based on the context.</p>")

        context_state = gr.State()
        model_choice_state = gr.State(value="BERT")  # Default model is BERT

        with gr.Row():
            with gr.Column(scale=11):  # Left panel for chatbot and question input (45%)
                chatbot = gr.Chatbot(label="Chatbot")
                question_input = gr.Textbox(label="Ask a Question", placeholder="Enter your question here...", lines=1)
                submit_btn = gr.Button("Submit Question")

            with gr.Column(scale=9):  # Right panel for setting context and instructions (55%)
                context_input = gr.Textbox(label="Set Context", placeholder="Enter the context here...", lines=4)
                set_context_btn = gr.Button("Set Context")
                clear_context_btn = gr.Button("Clear Context")

                # Model selection buttons
                model_selection = gr.Radio(choices=["BERT", "BART", "DistilBERT"], label="Select Model", value="BERT")
                status_message = gr.Markdown("")

                gr.Markdown("<strong>Instructions:</strong><br>1. Set a context.<br>2. Select the model (BERT, BART, or DistilBERT).<br>3. Ask questions based on the context.<br><br><strong>Note:</strong> <span class='note'>The BART model is pre-trained from Hugging Face. Credits to Hugging Face and the person who fine-tuned this model ('valhalla/bart-large-finetuned-squadv1')</span>")

        footer = gr.Markdown("<div class='footer'>Prepared by: Team EDITH</div>")

        def set_context(context):
            if not context.strip():
                return gr.update(), "Please enter a valid context.", None
            return gr.update(visible=False), "Context has been set. You can now ask questions.", context

        def clear_context():
            return gr.update(visible=True), "Context has been cleared. Please set a new context.", None

        def handle_question(question, history, context, model_choice):
            global tokenizer, model, device

            if not context:
                return history, "Please set the context before asking questions."
            if not question.strip():
                return history, "Please enter a valid question."

            # Load the selected model and tokenizer
            if model_choice == "BERT":
                tokenizer, model, device = load_bert_model_and_tokenizer()
                model_name = "BERT"
            elif model_choice == "BART":
                tokenizer, model, device = load_bart_model_and_tokenizer()
                model_name = "BART"
            elif model_choice == "DistilBERT":
                tokenizer, model, device = load_distilbert_model_and_tokenizer()
                model_name = "DistilBERT"

            answer = generate_answer(context, question)
            history = history + [[f"👤: {question}", f"🤖 ({model_name}): {answer}"]]  # Show the selected model with the answer
            return history, ""

        set_context_btn.click(set_context, inputs=context_input, outputs=[context_input, status_message, context_state])
        clear_context_btn.click(clear_context, inputs=None, outputs=[context_input, status_message, context_state])
        submit_btn.click(handle_question, inputs=[question_input, chatbot, context_state, model_selection], outputs=[chatbot, question_input])
        
        # Enable "Enter" key to trigger the "Submit" button
        question_input.submit(handle_question, inputs=[question_input, chatbot, context_state, model_selection], outputs=[chatbot, question_input])

    return demo

if __name__ == "__main__":
    demo = chatbot_interface()
    demo.launch(share=True)  # Enable public sharing
