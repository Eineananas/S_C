# -*- coding: utf-8 -*-
import json
import os
import re
import numpy as np
import pandas as pd


def parse_labels(result_str):
    if result_str is None:
        return []

    result_str = str(result_str).strip()

    if result_str.lower() in ["nan", "null", "none", ""]:
        return []

    matches = re.findall(r"\[([^\]]*)\]", result_str)

    if matches:
        inner = matches[-1]
    else:
        inner = result_str

    numbers = re.findall(r"\d+", inner)

    labels = []
    for num_str in numbers:
        try:
            labels.append(int(num_str))
        except (ValueError, TypeError):
            continue

    return labels


def r_precision_at_k(pred_labels, gold_set, k):
    if not gold_set:
        return 1.0

    r = len(gold_set)
    top_k_len = min(k, r)

    if top_k_len == 0:
        return 0.0

    top_k = pred_labels[:k]
    rel_in_topk = len(set(top_k) & gold_set)

    return rel_in_topk / top_k_len


def hit_at_k(pred_labels, gold_set, k):
    if not gold_set:
        return 1.0

    top_k = pred_labels[:k]

    if set(top_k) & gold_set:
        return 1.0

    return 0.0


def reciprocal_rank(pred_labels, gold_set):
    if not gold_set:
        return 1.0

    for rank_idx, pred in enumerate(pred_labels):
        if pred in gold_set:
            return 1.0 / (rank_idx + 1)

    return 0.0


all_results = []
extracted_data = []

folder_path = folder_path
output_file = os.path.join(folder_path, "Nature_Gemini_wrong.json")

if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename != "Nature_Gemini_wrong.json":
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                rp5_scores = []
                rp10_scores = []
                hit1_scores = []
                hit3_scores = []
                hit5_scores = []
                hit10_scores = []
                rr_scores = []

                total_instances = len(data)
                valid_instances = 0
                nan_error_count = 0

                print(f"\nProcessing file: {file_path}")

                for i, item in enumerate(data):
                    try:
                        if "ori_label" not in item:
                            continue

                        gold_labels = set(item["ori_label"])
                        result_str = item.get("result", None)
                        pred_labels = parse_labels(result_str)

                        if not pred_labels:
                            nan_error_count += 1

                            matched_item = (
                                item.copy()
                                if isinstance(item, dict)
                                else {"raw_item": str(item)}
                            )
                            matched_item["filename"] = filename
                            extracted_data.append(matched_item)

                            pred_labels = []

                        if gold_labels:
                            rp5 = r_precision_at_k(pred_labels, gold_labels, 5)
                            rp10 = r_precision_at_k(pred_labels, gold_labels, 10)

                            hit1 = hit_at_k(pred_labels, gold_labels, 1)
                            hit3 = hit_at_k(pred_labels, gold_labels, 3)
                            hit5 = hit_at_k(pred_labels, gold_labels, 5)
                            hit10 = hit_at_k(pred_labels, gold_labels, 10)

                            rr = reciprocal_rank(pred_labels, gold_labels)

                            rp5_scores.append(rp5)
                            rp10_scores.append(rp10)
                            hit1_scores.append(hit1)
                            hit3_scores.append(hit3)
                            hit5_scores.append(hit5)
                            hit10_scores.append(hit10)
                            rr_scores.append(rr)

                            valid_instances += 1

                    except Exception as e:
                        nan_error_count += 1
                        pred_labels = []

                        if "ori_label" in item and item["ori_label"]:
                            gold_labels = set(item["ori_label"])

                            rp5_scores.append(0.0)
                            rp10_scores.append(0.0)
                            hit1_scores.append(0.0)
                            hit3_scores.append(0.0)
                            hit5_scores.append(0.0)
                            hit10_scores.append(0.0)
                            rr_scores.append(0.0)

                            valid_instances += 1

                        matched_item = (
                            item.copy()
                            if isinstance(item, dict)
                            else {"raw_item": str(item)}
                        )
                        matched_item["filename"] = file
