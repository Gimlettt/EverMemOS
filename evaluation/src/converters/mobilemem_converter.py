"""
MobileMem Converter - convert MobileMem dataset to LoCoMo format.

MobileMem format:
- dialogue.json: flat list of 359 session entries, each with timestamp, location, and conversations
- demo_qa.json: dict keyed by category, with Q&A and ranking question formats

All sessions belong to a single user, so they map to one LoCoMo conversation with N sessions.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


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


@register_converter("mobilemem")
class MobileMemConverter(BaseConverter):
    """MobileMem dataset converter."""

    def get_input_files(self) -> Dict[str, str]:
        return {
            "dialogue": "dialogue.json",
            "qa": "demo_qa.json",
        }

    def get_output_filename(self) -> str:
        return "mobilemem_locomo_style.json"

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        print("Converting MobileMem to LoCoMo format...")

        # Load data
        with open(input_paths["dialogue"], "r", encoding="utf-8") as f:
            dialogues = json.load(f)
        with open(input_paths["qa"], "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        print(f"   Loaded {len(dialogues)} dialogue sessions")

        # Build single conversation (all sessions belong to one user)
        conversation = {
            "speaker_a": "User",
            "speaker_b": "Assistant",
        }

        for idx, entry in enumerate(dialogues):
            session_key = f"session_{idx}"

            # Convert timestamp
            conversation[f"{session_key}_date_time"] = format_locomo_timestamp(
                entry["timestamp"]
            )

            # Convert messages
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

        # Convert QA pairs
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

        # Ranking format categories (options assembled for the LLM)
        ranking_categories = [
            "personalized_prediction",
            "personalized_assistance",
        ]

        for category in ranking_categories:
            for item in qa_data.get(category, []):
                # Build options dict for all_options metadata
                options = {}
                for opt in item["options"]:
                    options[opt["option_char"]] = opt["text"]

                # correct_answer is a ranked list like ["C", "B", "A"]
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

        # Build LoCoMo format (single conversation entry)
        locomo_data = [
            {
                "conversation": conversation,
                "qa": qa_list,
            }
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_data, f, indent=2, ensure_ascii=False)

        total_questions = len(qa_list)
        category_counts = {}
        for qa in qa_list:
            cat = qa["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(
            f"   Saved 1 conversation with {len(dialogues)} sessions, {total_questions} questions"
        )
        for cat, count in category_counts.items():
            print(f"     {cat}: {count}")
