import re
from transformers import BertTokenizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def split_camel_case_in_sentences(sentences):
    """
    Splits camel case in sentences for better tokenization.

    Args:
        sentences (list[str]): List of sentences to process.

    Returns:
        list[str]: Sentences with camel case split.
    """

    def split_camel_case_word(word):
        return re.sub("([a-z0-9])([A-Z])", r"\1 \2", word)

    processed_sentences = []
    for sentence in sentences:
        processed_words = []
        for word in sentence.split():
            processed_words.extend(split_camel_case_word(word).split())
        processed_sentences.append(" ".join(processed_words))
    return processed_sentences


def preprocess(texts):
    """
    Preprocesses document texts for tokenization and reranking.

    Args:
        texts (list[str]): List of document texts to preprocess.

    Returns:
        list[str]: Preprocessed texts.
    """
    # Add spaces around punctuation
    texts = [re.sub(r"(?<=\w)([?.,!])(?!\s)", r" \1", text) for text in texts]

    # Replace specific characters and split camel case
    texts = [
        text.replace("_", " ").replace("-", " ").replace("#", " ").replace(".html", "").replace(".", " ")
        for text in texts
    ]
    texts = split_camel_case_in_sentences(texts)

    # Tokenize using BERT tokenizer
    texts = [tokenizer.tokenize(text) for text in texts]
    texts = [" ".join(word for word in text) for text in texts]
    return texts


def custom_predict(inputs, classifier, threshold=0.2, other_label="other"):
    """
    Custom classification prediction with thresholding.

    Args:
        inputs (list): Input features for classification.
        classifier: Pre-trained classifier.
        threshold (float): Minimum probability for classification.
        other_label (str): Label to assign if below threshold.

    Returns:
        list: Predicted labels.
    """
    y_pred = []
    for item in classifier.predict_proba(inputs):
        prob = max(item)
        index = item.argmax()
        if prob >= threshold:
            pred = classifier.classes_[index]
        else:
            pred = other_label
        y_pred.append(pred)
    return y_pred


def rerank_documents(documents, query_embedding, classifier=None, threshold=0.2):
    """
    Reranks retrieved documents based on a relevance score and optional classification.

    Args:
        documents (list[dict]): Retrieved documents with text and metadata.
        query_embedding (list[float]): Embedding of the user query.
        classifier: Optional classifier for domain-specific reranking.
        threshold (float): Threshold for classification relevance.

    Returns:
        list[dict]: Reranked list of documents.
    """
    # Extract texts and preprocess
    texts = [doc["text"] for doc in documents]
    preprocessed_texts = preprocess(texts)

    # If classifier is provided, apply classification-based reranking
    if classifier:
        predicted_tags = custom_predict(preprocessed_texts, classifier, threshold=threshold)
        for i, doc in enumerate(documents):
            doc["predicted_tag"] = predicted_tags[i]

    # Sort documents by relevance score and predicted tag (if available)
    reranked_documents = sorted(
        documents,
        key=lambda doc: (
            -doc.get("relevance_score", 0),  # Sort by relevance score in descending order
            doc.get("predicted_tag", ""),  # Secondary sort by predicted tag
        ),
    )
    return reranked_documents


def get_reranked_indices(documents, query_embedding, classifier=None, threshold=0.2):
    """
    Get indices of reranked documents for easy reference.

    Args:
        documents (list[dict]): Retrieved documents with text and metadata.
        query_embedding (list[float]): Embedding of the user query.
        classifier: Optional classifier for domain-specific reranking.
        threshold (float): Threshold for classification relevance.

    Returns:
        list[int]: Indices of documents in reranked order.
    """
    reranked_documents = rerank_documents(documents, query_embedding, classifier, threshold)
    return [documents.index(doc) for doc in reranked_documents]
