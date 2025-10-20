# Text-Summarization-Seq2Seq-

Overview

This project is a practical application of fine-tuning a pre-trained language model to perform Abstractive Text Summarization. The goal is to enable the model to read a long article and generate a short, accurate summary that captures the main ideas.

The project utilizes the popular Hugging Face libraries and PyTorch to train and evaluate the model on the well-known CNN/DailyMail dataset.

🌟 Key Features

Specialized Model: Uses the sshleifer/distilbart-cnn-12-6 model, a lightweight and efficient version of BART specifically fine-tuned for news summarization.

Fine-tuning: The code fine-tunes the model on a sample of the data to improve its performance.

Accurate Evaluation: The model's performance is evaluated using ROUGE metrics, the standard benchmark for summarization tasks.

Optimized Code: The code is optimized to run efficiently in memory-constrained environments like Google Colab, featuring memory-clearing techniques.

🛠️ Technologies and Libraries Used

PyTorch: The primary deep learning framework.

Hugging Face Transformers: For loading models, tokenizers, and executing the training loop.

Hugging Face Datasets: For easily loading and processing the dataset.

Hugging Face Evaluate: For calculating performance metrics (ROUGE).

🚀 How to Set Up and Run

To ensure a smooth experience and avoid the compatibility issues we faced previously, please follow these steps precisely:

Step 1: Start with a Clean Environment (Crucial)

Create a brand new Notebook in Google Colab or Kaggle.

Make sure to enable the GPU from the Notebook's settings.

In Colab: Go to Runtime -> Change runtime type -> Select GPU.

Step 2: Run the Code Directly

Copy the entire code from the summarization_pytorch.py file.

Paste the code into a single cell in the new Notebook you created.

Run the cell. The new environment will be compatible, and the code will download everything and run successfully.

Note: You typically do not need to run any pip install commands in a new Colab/Kaggle environment, as most libraries are pre-installed.

📄 Code Explanation

The code is divided into 5 main logical steps:

Load Data: A small sample of the cnn_dailymail dataset is loaded to save time and memory.

Load Model: The pre-trained model and tokenizer are loaded from the Hugging Face Hub.

Preprocessing: The articles and summaries are prepared and converted into numbers (tokens) to be suitable for the model.

Training: The Seq2SeqTrainer class is used to fine-tune the model on the processed data.

Evaluation and Generation: After training is complete, the model is evaluated on data it has not seen before, and a practical example is shown to generate a summary for a random article.

📊 Expected Results

Upon successful execution of the code, you will see the following outputs:

A progress bar for the training process, showing the Loss value decreasing.

A table with the ROUGE score results (e.g., rouge1, rouge2, rougeL) after training.

An example showing:

The Original Article.

The Generated Summary produced by the model.

The Actual Summary written by a human.
