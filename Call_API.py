import json
import os
import time
from typing import Any, Dict, List, Tuple
from google import genai
from google.genai import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GOOGLE_APPLICATION_CREDENTIALS"


class RecruitmentExampleGenerator:

    def __init__(
        self, project_id: str, location: str, model_id: str, prompt_file: str
    ):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        self.model_id = model_id
        self.system_prompt = self._load_system_prompt(prompt_file)

    def _load_system_prompt(self, prompt_file: str) -> str:
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]

        for enc in encodings:
            try:
                with open(prompt_file, "r", encoding=enc) as f:
                    prompt = f.read().strip()
                print(f" Loaded prompt with {enc}")
                return prompt
            except UnicodeDecodeError:
                print(f"Failed with {enc}")
                continue

        with open(
            prompt_file, "r", encoding="utf-8", errors="ignore"
        ) as f:
            prompt = f.read().strip()
        print(" Used utf-8 with errors=ignore")
        return prompt

    def call_api(
        self, query: str, skill_text: str, max_retries: int = 3
    ) -> str:
        user_content = f"{query}\n\nSkill Label Space:\n{skill_text}"

        print(f"API: query({len(query)})+skills({len(skill_text)})")

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
                    print(f" API OK: {content[:200]}...")
                    return content
                else:
                    raise Exception("Empty response received from Gemini API")

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(attempt * 3 + 3)
                else:
                    break

        raise Exception("Gemini API failed after retries")

    def load_data(
        self, file_path: str
    ) -> Tuple[List[str], List[List[int]], List[List[Dict]]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = []
        ori_labels = []
        rerank_results_list = []

        for idx, entry in enumerate(data):
            queries.append(entry.get("query", ""))
            label_str = entry.get("ori_label", "")
            labels = []

            if isinstance(label_str, str):
                label_str_clean = label_str.strip()
                if label_str_clean: 
                    try:
                        labels = json.loads(label_str_clean)
                    except json.JSONDecodeError as e:
                        print(
                            f"fAILURE: {e}"
                        )
                        labels = []
            elif isinstance(label_str, (list, tuple)):
                labels = label_str

            parsed_ori_label = []
            if isinstance(labels, list):
                for x in labels:
                    try:
                        parsed_ori_label.append(int(x))
                    except (ValueError, TypeError):
                        continue

            ori_labels.append(parsed_ori_label)
            rerank_results_list.append(entry.get("rerank_results", []))

        print(f" Loaded {len(queries)} samples from {file_path}")
        return queries, ori_labels, rerank_results_list

    def load_skills(self, skill_file: str) -> List[str]:
        with open(skill_file, "r", encoding="utf-8") as f:
            skills = [line.strip() for line in f if line.strip()]
        print(f" Loaded {len(skills)} skills (0-{len(skills)-1})")
        return skills

    def get_skill_text(
        self, rerank_results: List[Dict], skills: List[str], top_k: int
    ) -> str:
        indices = []
        for result in rerank_results[:top_k]:
            doc_id = str(result.get("document_content", "")).strip()
            if doc_id.isdigit():
                idx = int(doc_id) - 1
                if 0 <= idx < len(skills):
                    indices.append(idx)

        unique = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)

        skill_texts = [skills[idx] for idx in unique]
        return " / ".join(skill_texts)

    def process_file(
        self, input_file: str, skill_file: str, output_file: str, top_k: int
    ):
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(input_file)} (Top-K: {top_k})")
        print(f"{'='*60}")

        if not os.path.exists(input_file):
            print(" Input file not found")
            return

        queries, ori_labels, rerank_results_list = self.load_data(input_file)
        skills = self.load_skills(skill_file)

        results = []
        for i, (query, ori_label, rerank) in enumerate(
            zip(queries, ori_labels, rerank_results_list)
        ):
            try:
                print(f"[{i+1}/{len(queries)}] Processing...")
                skill_text = self.get_skill_text(rerank, skills, top_k)

                if not skill_text:
                    skill_text = "No skills"

                api_result = self.call_api(query, skill_text)

                results.append(
                    {
                        "query": query,
                        "ori_label": ori_label,
                        "skill_text": skill_text,
                        "result": api_result,
                    }
                )

            except Exception as e:
                print(f" Query {i+1} Error: {e}")
                results.append(
                    {
                        "query": query,
                        "ori_label": ori_label,
                        "result": str(e),
                        "error": str(e),
                    }
                )

            time.sleep(1)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(
            f" Saved {len([r for r in results if 'error' not in r])}/{len(results)} to {output_file}"
        )


def main():
    CONFIG = {
        "PROJECT_ID": "PROJECT_ID",
        "LOCATION": "global",
        "MODEL_ID": "gemini-2.5-flash",
        "PROMPT_PATH": "PROMPT_PATH",
        "SKILL_FILE": "SKILL_FILE",
        "BASE_PATH": "BASE_PATH",
    }

    generator = RecruitmentExampleGenerator(
        project_id=CONFIG["PROJECT_ID"],
        location=CONFIG["LOCATION"],
        model_id=CONFIG["MODEL_ID"],
        prompt_file=CONFIG["PROMPT_PATH"],
    )
    top_k = top_k
    input_file = os.path.join(
        CONFIG["BASE_PATH"], f"input.json"
    )
    output_file = os.path.join(
        CONFIG["BASE_PATH"], f"out.json"
    )
    generator.process_file(
        input_file, CONFIG["SKILL_FILE"], output_file, top_k
    )


if __name__ == "__main__":
    main()
