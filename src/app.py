import gradio as gr
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTableQuestionAnswering,
    AutoTokenizer,
    pipeline,
)

model_tapex = "microsoft/tapex-large-finetuned-wtq"
tokenizer_tapex = AutoTokenizer.from_pretrained(model_tapex)
model_tapex = AutoModelForSeq2SeqLM.from_pretrained(model_tapex)
pipe_tapex = pipeline(
    "table-question-answering", model=model_tapex, tokenizer=tokenizer_tapex
)

model_tapas = "google/tapas-large-finetuned-wtq"
tokenizer_tapas = AutoTokenizer.from_pretrained(model_tapas)
model_tapas = AutoModelForTableQuestionAnswering.from_pretrained(model_tapas)
pipe_tapas = pipeline(
    "table-question-answering", model=model_tapas, tokenizer=tokenizer_tapas
)


def process(query, file, correct_answer, rows=20):
    table = pd.read_csv(file.name, header=0).astype(str)
    table = table[:rows]
    result_tapex = pipe_tapex(table=table, query=query)
    result_tapas = pipe_tapas(table=table, query=query)
    return result_tapex["answer"], result_tapas["answer"], correct_answer


# Inputs
query_text = gr.Text(label="Enter a question")
input_file = gr.File(label="Upload a CSV file", type="file")
rows_slider = gr.Slider(label="Number of rows")

# Output
answer_text_tapex = gr.Text(label="TAPEX answer")
answer_text_tapas = gr.Text(label="TAPAS answer")

description = "This Space lets you ask questions on CSV documents with Microsoft [TAPEX-Large](https://huggingface.co/microsoft/tapex-large-finetuned-wtq) and Google [TAPAS-Large](https://huggingface.co/google/tapas-large-finetuned-wtq). \
Both have been fine-tuned on the [WikiTableQuestions](https://huggingface.co/datasets/wikitablequestions) dataset. \n\n\
A sample file with football statistics is available in the repository: \n\n\
* Which team has the most wins? Answer: Manchester City FC\n\
* Which team has the most wins: Chelsea, Liverpool or Everton? Answer: Liverpool\n\
* Which teams have scored less than 40 goals? Answer: Cardiff City FC, Fulham FC, Brighton & Hove Albion FC, Huddersfield Town FC\n\
* What is the average number of wins? Answer: 16 (rounded)\n\n\
You can also upload your own CSV file. Please note that maximum sequence length for both models is 1024 tokens, \
so you may need to limit the number of rows in your CSV file. Chunking is not implemented yet."

iface = gr.Interface(
    theme="huggingface",
    description=description,
    layout="vertical",
    fn=process,
    inputs=[query_text, input_file, rows_slider],
    outputs=[answer_text_tapex, answer_text_tapas],
    examples=[
        ["Which team has the most wins?", "default_file.csv", 20],
        [
            "Which team has the most wins: Chelsea, Liverpool or Everton?",
            "default_file.csv",
            20,
        ],
        ["Which teams have scored less than 40 goals?", "default_file.csv", 20],
        ["What is the average number of wins?", "default_file.csv", 20],
    ],
    allow_flagging="never",
)

iface.launch()
