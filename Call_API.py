import json
import os
import time
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types


os.environ["API_CREDENTIALS"] = "YOUR_CREDENTIALS"


class ExampleGenerator:
    def __init__(
        self,
        project_id: str,
        location: str,
        model_id: str,
        prompt_file: str,
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        self.model_id = model_id
        self.system_prompt = self.load_system_prompt(prompt_file)

    def load_system_prompt(self, prompt_file: str) -> str:
        encodings = [
            "utf-8",
            "utf-8-sig",
            "gbk",
            "gb2312",
            "latin1",
        ]

        for encoding in encodings:
            try:
                with open(prompt_file, "r", encoding=encoding) as file:
                    prompt = file.read().strip()

                print(f"Loaded prompt using {encoding}.")
                return prompt

            except UnicodeDecodeError:
                print(f"Could not decode prompt using {encoding}.")
                continue

        with open(
            prompt_file,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:
            prompt = file.read().strip()

        print("Loaded prompt using utf-8 with ignored decoding errors.")
        return prompt

    def call_api(
        self,
        query: str,
        candidate_text: str,
        max_retries: int = 3,
    ) -> str:
        user_content = (
            f"{query}\n\n"
            f"Candidate Label Space:\n{candidate_text}"
        )

        print(
            f"Calling API with query length {len(query)} "
            f"and candidate-text length {len(candidate_text)}."
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                    ),
                )

                if response.text:
                    content = response.text.strip()
                    print(f"API call succeeded: {content[:200]}...")
                    return content

                raise RuntimeError("Received an empty API response.")

            except Exception as error:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed: "
                    f"{error}"
                )

                if attempt < max_retries - 1:
                    time.sleep(attempt * 3 + 3)

        raise RuntimeError("API call failed after all retry attempts.")

    def load_data(
        self,
        file_path: str,
    ) -> Tuple[List[str], List[List[int]], List[List[Dict[str, Any]]]]:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        queries = []
        reference_labels = []
        ranked_candidates_list = []

        for entry in data:
            queries.append(entry.get("query", ""))

            raw_labels = entry.get(
                "reference_label",
                entry.get("ori_label", ""),
            )

            labels = []

            if isinstance(raw_labels, str):
                cleaned_labels = raw_labels.strip()

                if cleaned_labels:
                    try:
                        labels = json.loads(cleaned_labels)

                    except json.JSONDecodeError as error:
                        print(
                            f"Could not parse reference labels: {error}"
                        )
                        labels = []

            elif isinstance(raw_labels, (list, tuple)):
                labels = raw_labels

            parsed_labels = []

            if isinstance(labels, list):
                for label in labels:
                    try:
                        parsed_labels.append(int(label))
                    except (ValueError, TypeError):
                        continue

            reference_labels.append(parsed_labels)

            ranked_candidates_list.append(
                entry.get(
                    "ranked_candidates",
                    entry.get("rerank_results", []),
                )
            )

        print(f"Loaded {len(queries)} examples from {file_path}.")

        return queries, reference_labels, ranked_candidates_list

    def load_candidate_space(self, candidate_file: str) -> List[str]:
        with open(candidate_file, "r", encoding="utf-8") as file:
            candidates = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(
            f"Loaded {len(candidates)} candidate labels "
            f"(indices 0 to {len(candidates) - 1})."
        )

        return candidates

    def get_candidate_text(
        self,
        ranked_candidates: List[Dict[str, Any]],
        candidates: List[str],
        top_k: int,
    ) -> str:
        indices = []

        for result in ranked_candidates[:top_k]:
            document_id = str(
                result.get("document_content", "")
            ).strip()

            if document_id.isdigit():
                index = int(document_id) - 1

                if 0 <= index < len(candidates):
                    indices.append(index)

        unique_indices = []
        seen_indices = set()

        for index in indices:
            if index not in seen_indices:
                unique_indices.append(index)
                seen_indices.add(index)

        selected_candidates = [
            candidates[index]
            for index in unique_indices
        ]

        return " / ".join(selected_candidates)

    def process_file(
        self,
        input_file: str,
        candidate_file: str,
        output_file: str,
        top_k: int,
    ):
        print("\n" + "=" * 60)
        print(
            f"Processing file: {os.path.basename(input_file)} "
            f"(Top-K: {top_k})"
        )
        print("=" * 60)

        if not os.path.exists(input_file):
            print(f"Input file was not found: {input_file}")
            return

        queries, reference_labels, ranked_candidates_list = (
            self.load_data(input_file)
        )

        candidates = self.load_candidate_space(candidate_file)
        results = []

        for index, (query, reference_label, ranked_candidates) in enumerate(
            zip(
                queries,
                reference_labels,
                ranked_candidates_list,
            )
        ):
            try:
                print(
                    f"Processing example "
                    f"{index + 1}/{len(queries)}."
                )

                candidate_text = self.get_candidate_text(
                    ranked_candidates,
                    candidates,
                    top_k,
                )

                if not candidate_text:
                    candidate_text = "No candidate labels available."

                api_result = self.call_api(
                    query=query,
                    candidate_text=candidate_text,
                )

                results.append(
                    {
                        "query": query,
                        "reference_label": reference_label,
                        "candidate_text": candidate_text,
                        "result": api_result,
                    }
                )

            except Exception as error:
                print(
                    f"Error while processing example {index + 1}: "
                    f"{error}"
                )

                results.append(
                    {
                        "query": query,
                        "reference_label": reference_label,
                        "result": str(error),
                        "error": str(error),
                    }
                )

            time.sleep(1)

        output_directory = os.path.dirname(output_file)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                ensure_ascii=False,
                indent=2,
            )

        successful_count = len(
            [
                result
                for result in results
                if "error" not in result
            ]
        )

        print(
            f"Saved {successful_count}/{len(results)} "
            f"successfully processed examples to {output_file}."
        )


def main():
    config = {
        "PROJECT_ID": "YOUR_PROJECT_ID",
        "LOCATION": "YOUR_REGION",
        "MODEL_ID": "YOUR_MODEL_ID",
        "PROMPT_PATH": "/path/to/prompt.txt",
        "CANDIDATE_FILE": "/path/to/candidate_space.txt",
        "INPUT_FILE": "/path/to/input.json",
        "OUTPUT_FILE": "/path/to/output.json",
        "TOP_K": 50,
    }

    generator = ExampleGenerator(
        project_id=config["PROJECT_ID"],
        location=config["LOCATION"],
        model_id=config["MODEL_ID"],
        prompt_file=config["PROMPT_PATH"],
    )

    generator.process_file(
        input_file=config["INPUT_FILE"],
        candidate_file=config["CANDIDATE_FILE"],
        output_file=config["OUTPUT_FILE"],
        top_k=config["TOP_K"],
    )


if __name__ == "__main__":
    main()
