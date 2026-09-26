import gc
import json
import logging
import os
import re
from typing import Any, Dict, List
import numpy as np
import torch
from pymilvus.model.reranker import BGERerankFunction
from sentence_transformers import SentenceTransformer


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EMBEDDING_MODEL_NAME = "your-embedding-model"
RERANKER_MODEL_NAME = "your-reranker-model"

VAULT_IDS = ["corpus_a"]
DATASET_NAMES = ["dataset_a", "dataset_b", "dataset_c"]
RETRIEVAL_SIZES = [70, 100, 150, 200, 300]

VAULT_PATH_TEMPLATE = "/path/to/corpus/corpus_{vault_id}.txt"
INPUT_PATH_TEMPLATE = "/path/to/retrieval_outputs/{dataset}_{vault_id}_top{k}.json"
OUTPUT_PATH_TEMPLATE = (
    "/path/to/reranking_outputs/{dataset}_{vault_id}_top{k}_reranked.json"
)


class MemoryEfficientRetrievalSystem:
    def __init__(self, embedding_model_name: str, reranker_model_name: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

        self.model = None
        self.reranker = None
        self.corpus_embeddings = None
        self.corpus_content = None

    def load_embedding_model(self):
        if self.model is None:
            self.model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )

        return self.model

    def load_reranker(self):
        if self.reranker is None:
            self.reranker = BGERerankFunction(
                model_name=self.reranker_model_name,
                device=self.device,
            )

        return self.reranker

    def unload_models(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.reranker is not None:
            del self.reranker
            self.reranker = None

        self.clean_memory()

    def clean_memory(self):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_corpus(self, corpus_path: str, batch_size: int = 64):
        logger.info(f"Loading corpus: {corpus_path}")

        with open(corpus_path, "r", encoding="utf-8") as file:
            self.corpus_content = [
                line.strip()
                for line in file
                if line.strip()
            ]

        logger.info(
            f"Encoding {len(self.corpus_content)} corpus documents in batches."
        )

        embedding_model = self.load_embedding_model()
        embedding_batches = []

        for start_index in range(0, len(self.corpus_content), batch_size):
            batch_content = self.corpus_content[
                start_index:start_index + batch_size
            ]

            batch_embeddings = embedding_model.encode(
                batch_content,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=False,
            )

            embedding_batches.append(batch_embeddings)

            if (start_index // batch_size) % 10 == 0:
                processed_count = min(
                    start_index + batch_size,
                    len(self.corpus_content),
                )

                logger.info(
                    f"Encoded {processed_count}/{len(self.corpus_content)} documents."
                )

                self.clean_memory()

        all_embeddings = np.vstack(embedding_batches)

        self.corpus_embeddings = torch.tensor(
            all_embeddings,
            device="cpu",
        )

        logger.info(
            f"Corpus encoding completed. Shape: {self.corpus_embeddings.shape}"
        )

    def extract_document_id(self, content: str) -> str:
        match = re.search(r"Item\s+(\d+):", content)

        if match:
            return match.group(1)

        return content[:50] + "..." if len(content) > 50 else content

    @staticmethod
    def split_spans(span_text: str) -> List[str]:
        if not span_text:
            return []

        cleaned_text = re.sub(
            r"^[\[\s]+|[\]\s]+$",
            "",
            span_text,
        )

        if not cleaned_text:
            return []

        return [
            item.strip()
            for item in cleaned_text.split(";")
            if item.strip()
        ]

    def retrieve_context_indices(
        self,
        query_text: str,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        if self.corpus_embeddings is None or not self.corpus_content:
            return
