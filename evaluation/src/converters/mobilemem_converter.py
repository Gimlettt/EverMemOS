"""
MobileMem Converter - convert MobileMem dataset to LoCoMo format.

MobileMem format:
- demo_dataset/{user}/ directories, each containing:
  - A processed dialogue JSON (with captions/location embedded in conversation)
  - Optionally a demo_qa.json with questions

Each user becomes a separate LoCoMo conversation entry.
Kevin is always index 0 to preserve backward compatibility with existing results.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter

# User directory → dialogue filename mapping
# Kevin first to ensure mobilemem_0 = kevin (backward compatible)
USER_DIALOGUE_FILES = {
    "kevin": "processed_dialogue_w_cap.json",
    "hao": "hao_processed_dialogue_with_captions.json",
    "jr": "jr_processed_dialogue_with_captions.json",
    "kaiao": "kaiao_processed_dialogue_with_captions.json",
    "lv": "lv_processed_dialogue_with_captions.json",
    "meng": "meng_processed_dialogue_with_captions.json",
    "xinyi": "xinyi_processed_dialogue_with_captions.json",
}


def format_locomo_timestamp(ts_str: str) -> str:
    """
    Convert MobileMem timestamp to LoCoMo format.

    Input:  "2020:09:19 17:11:49"
    Output: "5:11 pm on 19 September, 2020"
    """
    dt = datetime.strptime(ts_str, "%Y:%m:%d %H:%M:%S")
    hour = str(int(dt.strftime("%I")))
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    day = str(dt.day)
    month = dt.strftime("%B")
    year = str(dt.year)
    return f"{hour}:{minute} {ampm} on {day} {month}, {year}"


def convert_dialogues_to_conversation(dialogues: list) -> dict:
    """Convert a list of dialogue sessions to a LoCoMo conversation dict."""
    conversation = {
        "speaker_a": "User",
        "speaker_b": "Assistant",
    }

    for idx, entry in enumerate(dialogues):
        session_key = f"session_{idx}"
        conversation[f"{session_key}_date_time"] = format_locomo_timestamp(
            entry["timestamp"]
        )

        messages = []
        for msg_idx, msg in enumerate(entry["conversations"]):
            messages.append(
                {
                    "speaker": msg["speaker"],
                    "text": msg["text"],
                    "dia_id": f"D{idx}:{msg_idx}",
                }
            )
        conversation[session_key] = messages

    return conversation


def convert_qa_data(qa_data: dict) -> list:
    """Convert MobileMem QA data to LoCoMo QA list."""
    qa_list = []

    # Q&A format categories
    qa_categories = [
        "specific_event_retrieval",
        "information_extraction",
        "summarization_n_comparison",
        "multimodal_recall",
        "habit_preference_identification",
        "habit_preference_tracking",
        "reasoning_for_pref_behaviors",
    ]

    for category in qa_categories:
        for item in qa_data.get(category, []):
            qa_list.append(
                {
                    "question": item["question"],
                    "answer": item["correct_answer"],
                    "category": category,
                    "evidence": [],
                    "evidence_ids": item.get("evidence_ids", []),
                }
            )

    # Ranking format categories
    ranking_categories = [
        "personalized_prediction",
        "personalized_assistance",
    ]

    for category in ranking_categories:
        for item in qa_data.get(category, []):
            options = {}
            for opt in item["options"]:
                options[opt["option_char"]] = opt["text"]

            qa_list.append(
                {
                    "question": item["question"],
                    "answer": ", ".join(item["correct_answer"]),
                    "category": category,
                    "evidence": [],
                    "all_options": options,
                    "is_ranking": True,
                }
            )

    return qa_list


@register_converter("mobilemem")
class MobileMemConverter(BaseConverter):
    """MobileMem multi-user dataset converter."""

    def get_input_files(self) -> Dict[str, str]:
        # Point to the demo_dataset directory
        return {
            "demo_dataset": "demo_dataset",
        }

    def get_output_filename(self) -> str:
        return "mobilemem_locomo_style.json"

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        print("Converting MobileMem to LoCoMo format...")

        demo_dataset_dir = Path(input_paths["demo_dataset"])
        locomo_data = []

        for user, dialogue_file in USER_DIALOGUE_FILES.items():
            user_dir = demo_dataset_dir / user
            dialogue_path = user_dir / dialogue_file
            qa_path = user_dir / "demo_qa.json"

            if not dialogue_path.exists():
                print(f"   Skipping {user}: {dialogue_file} not found")
                continue

            # Load dialogue
            with open(dialogue_path, "r", encoding="utf-8") as f:
                dialogues = json.load(f)

            total_msgs = sum(len(e["conversations"]) for e in dialogues)

            # Convert dialogue to LoCoMo conversation
            conversation = convert_dialogues_to_conversation(dialogues)

            # Load QA if available
            qa_list = []
            if qa_path.exists():
                with open(qa_path, "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
                qa_list = convert_qa_data(qa_data)

            locomo_data.append(
                {
                    "user_name": user,
                    "conversation": conversation,
                    "qa": qa_list,
                }
            )

            print(
                f"   {user}: {len(dialogues)} sessions, {total_msgs} messages, {len(qa_list)} questions"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_data, f, indent=2, ensure_ascii=False)

        total_convs = len(locomo_data)
        total_questions = sum(len(entry["qa"]) for entry in locomo_data)
        print(f"   Saved {total_convs} conversations, {total_questions} total questions")
