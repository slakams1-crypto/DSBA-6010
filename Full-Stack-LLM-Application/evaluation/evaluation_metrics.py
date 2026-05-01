import evaluate

def calculate_exact_match(qa_data, model_predictions):
    """
    Calculates the Exact Match (EM) score for question answering.

    Args:
        qa_data (list): A list of dictionaries representing the ground truth QA dataset.
                           Each dictionary should have 'id' and 'answers' (with 'text').
        model_predictions (list): A list of dictionaries representing the model's predictions.
                                Each dictionary should have 'id' and 'prediction_text'.

    Returns:
        dict: The results from the exact_match metric computation.
    """
    exact_match_metric = evaluate.load("exact_match")

    predictions_list = []
    references_list = []

    qa_data_dict = {item['id']: item['answers'][0]['text'] for item in qa_data}
    model_predictions_dict = {item['id']: item['prediction_text'] for item in model_predictions}

    common_ids = sorted(list(qa_data_dict.keys() & model_predictions_dict.keys()))

    for qid in common_ids:
        references_list.append(qa_data_dict[qid])
        predictions_list.append(model_predictions_dict[qid])

    print(f"\nReferences: {references_list}")
    print(f"Predictions: {predictions_list}")

    results = exact_match_metric.compute(predictions=predictions_list, references=references_list)

    return results

def calculate_rouge_score(predictions, references):
    """
    Calculates ROUGE scores.

    Args:
        predictions (list): A list of predicted texts.
        references (list): A list of reference texts.

    Returns:
        dict: The ROUGE scores.
    """
    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(predictions=predictions, references=references)
    return results

def calculate_bleu_score(predictions, references):
    """
    Calculates BLEU scores.

    Args:
        predictions (list): A list of predicted texts.
        references (list): A list of reference texts.

    Returns:
        dict: The BLEU scores.
    """
    bleu_metric = evaluate.load("bleu")
    # BLEU metric expects references to be a list of lists
    results = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    return results

def calculate_bert_score(predictions, references, lang="en"):
    """
    Calculates BERTScore.

    Args:
        predictions (list): A list of predicted texts.
        references (list): A list of reference texts.
        lang (str): Language for BERTScore (default is "en").

    Returns:
        dict: The BERTScore results.
    """
    bertscore_metric = evaluate.load("bertscore")
    results = bertscore_metric.compute(predictions=predictions, references=references, lang=lang)
    return results

def calculate_squad_v2_score(qa_data, model_predictions):
    """
    Calculates SQuAD v2 F1 and Exact Match scores for question answering.

    Args:
        qa_data (list): A list of dictionaries representing the ground truth QA dataset.
                           Each dictionary should have 'id' and 'answers' (with 'text').
        model_predictions (list): A list of dictionaries representing the model's predictions.
                                Each dictionary should have 'id' and 'prediction_text'.

    Returns:
        dict: The results from the squad_v2 metric computation (including F1 and Exact Match).
    """
    squad_metric = evaluate.load("squad_v2")

    # Format predictions for squad_v2 metric
    formatted_predictions = [
        {"id": item["id"], "prediction_text": item["prediction_text"], "no_answer_probability": 0.0}
        for item in model_predictions
    ]

    # Format references for squad_v2 metric
    formatted_references = [
        {
            "id": item["id"],
            "answers": {
                "answer_start": [item["answers"][0]["answer_start"]],
                "text": [item["answers"][0]["text"]]
            }
        }
        for item in qa_data
    ]

    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )
    return results
