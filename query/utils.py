import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from tiktoken import get_encoding
from ollama import OllamaClient


def get_num_tokens(text):
    """
    Calculate the number of tokens in a text using tiktoken.

    Args:
        text (str): Input text.

    Returns:
        int: Number of tokens.
    """
    enc = get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim(text, max_context_length):
    """
    Trim the text to fit within the max_context_length.

    Args:
        text (str): Input text.
        max_context_length (int): Maximum allowed token length.

    Returns:
        str: Trimmed text.
    """
    enc = get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])


def get_client(llm):
    """
    Initialize the appropriate LLM client based on the specified model.

    Args:
        llm (str): Name of the LLM model.

    Returns:
        OllamaClient: Client for interacting with the specified LLM API.
    """
    if "llama3.2" in llm.lower():
        # Local Llama3.2 setup
        base_url = os.getenv("LLAMA3_2_API_BASE", "http://localhost:11434")
        api_key = os.getenv("LLAMA3_2_API_KEY", None)  # Optional API key
        return OllamaClient(base_url=base_url, api_key=api_key)
    elif "ollama" in llm.lower():
        # Ollama setup
        base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        api_key = os.getenv("OLLAMA_API_KEY", None)  # Optional API key
        return OllamaClient(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM model: {llm}")


def execute_bash(command):
    """
    Execute a bash command and return the result.

    Args:
        command (str): Bash command to execute.

    Returns:
        subprocess.CompletedProcess: Result of the bash command execution.
    """
    results = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return results


def predict(inputs, preprocess_fnc, tokenizer, model, label_encoder, device="cpu", threshold=0.0):
    """
    Perform prediction using a classification model and assign labels based on threshold.

    Args:
        inputs (list): List of input texts.
        preprocess_fnc (function): Preprocessing function for inputs.
        tokenizer: Tokenizer for the model.
        model: Pre-trained classification model.
        label_encoder: Encoder for mapping labels to indices and vice versa.
        device (str): Device to run the model on ("cpu" or "cuda").
        threshold (float): Minimum probability threshold to assign a label.

    Returns:
        tuple: Predicted labels and probabilities for each input.
    """
    # Preprocess inputs and tokenize
    model.eval()
    inputs = [preprocess_fnc(item) for item in inputs]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    y_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

    # Assign labels based on probabilities
    labels = []
    for prob in y_probs:
        max_prob = np.max(prob)
        if max_prob < threshold:
            labels.append("other")
        else:
            labels.append(label_encoder.inverse_transform([prob.argmax()])[0])
    return labels, y_probs
