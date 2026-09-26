import json
import re

import torch
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "your-embedding-model"
VAULT_IDS = [0]

VAULT_PATH_TEMPLATE = "/path/to/vault/vault_{vault_id}.txt"
INPUT_PATH_TEMPLATE = "/path/to/input/input_{dataset}.json"
OUTPUT_PATH_TEMPLATE = "/path/to/output/{dataset}_{vault_id}_top{k}.json"

DATASETS = ["dataset_a", "dataset_b", "dataset_c"]
TOP_K_VALUES = [100, 150, 200, 300]


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


def get_relevant_context_indices(
    query_text,
    vault_embeddings_tensor,
    top_k,
    model,
):
    if vault_embeddings_tensor.nelement() == 0:
        return []

    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
    )

    query_tensor = torch.tensor(
        query_embedding,
        device=DEVICE,
        dtype=vault_embeddings_tensor.dtype,
    )

    cosine_scores = torch.matmul(vault_embeddings_tensor, query_tensor)
    effective_top_k = min(top_k, cosine_scores.size(0))

    return torch.topk(
        cosine_scores,
        k=effective_top_k,
    ).indices.tolist()


def split_span(span_text):
    if not span_text:
        return []

    cleaned_text = re.sub(r"^[\[\s]+|[\]\s]+$", "", span_text)

    if not cleaned_text:
        return []

    return [
        item.strip()
        for item in cleaned_text.split(";")
        if item.strip()
    ]


def get_item_key(context_text):
    if ":" in context_text:
        return context_text.split(":", 1)[0].strip()

    return context_text.strip()


def generate_retrieval_results(
    input_json_path,
    output_json_path,
    top_k,
    vault_embeddings_tensor,
    vault_content,
    vault_keys,
    model,
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for item in data:
        query_text = item.get("query", "")
        original_label = item.get("ori_label", "")
        span_text = item.get("skill_span", "")

        span_list = split_span(span_text)

        query_indices = get_relevant_context_indices(
            query_text,
            vault_embeddings_tensor,
            top_k,
            model,
        )

        span_indices = []

        for span in span_list:
            span_indices.extend(
                get_relevant_context_indices(
                    span,
                    vault_embeddings_tensor,
                    10,
                    model,
                )
            )

        seen_keys = set()
        selected_indices = []

        for index in span_indices:
            item_key = vault_keys[index]

            if item_key not in seen_keys:
                seen_keys.add(item_key)
                selected_indices.append(index)

        if len(selected_indices) > top_k:
            final_indices = selected_indices

        elif len(selected_indices) < top_k:
            final_indices = selected_indices.copy()

            for index in query_indices:
                item_key = vault_keys[index]

                if item_key not in seen_keys:
                    seen_keys.add(item_key)
                    final_indices.append(index)

                if len(final_indices) == top_k:
                    break

        else:
            final_indices = selected_indices

        retrieved_context = [
            vault_content[index].strip()
            for index in final_indices
        ]

        query_only_context = [
            vault_content[index].strip()
            for index in query_indices
        ]

        results.append(
            {
                "query": query_text,
                "skill_span": span_text,
                "ori_label": original_label,
                "output": retrieved_context,
                "output2": query_only_context,
            }
        )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


print(f"Loading embedding model on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

for vault_id in VAULT_IDS:
    vault_path = VAULT_PATH_TEMPLATE.format(vault_id=vault_id)

    print(f"\nProcessing retrieval corpus: {vault_path}")

    with open(vault_path, "r", encoding="utf-8") as vault_file:
        vault_content = vault_file.readlines()

    print("Creating unique item keys...")
    vault_keys = [get_item_key(line) for line in vault_content]

    print("Encoding retrieval corpus...")
    vault_embeddings_np = model.encode(
        vault_content,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    vault_embeddings_tensor = torch.tensor(
        vault_embeddings_np,
        device=DEVICE,
    )

    for dataset_name in DATASETS:
        for top_k in TOP_K_VALUES:
            input_json_path = INPUT_PATH_TEMPLATE.format(
                dataset=dataset_name
            )

            output_json_path = OUTPUT_PATH_TEMPLATE.format(
                dataset=dataset_name,
                vault_id=vault_id,
                k=top_k,
            )

            print("-" * 40)
            print(f"Input file: {input_json_path}")
            print(f"Output file: {output_json_path}")
            print(f"Target Top-K: {top_k}")

            generate_retrieval_results(
                input_json_path=input_json_path,
                output_json_path=output_json_path,
                top_k=top_k,
                vault_embeddings_tensor=vault_embeddings_tensor,
                vault_content=vault_content,
                vault_keys=vault_keys,
                model=model,
            )
