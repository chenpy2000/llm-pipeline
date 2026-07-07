import main_SFT_QA_CoT as sft


def configure_coding_sft():
    sft.STAGE_NAME = "sft_coding"
    sft.TOKENIZED_DATA_DIR = "./data/tokenized_sft_coding"

    sft.SFT_SOURCES = [
        {
            "label": "opencodeinstruct",
            "dataset_id": "nvidia/OpenCodeInstruct",
            "split": "train",
            "user_columns": ["input"],
            "assistant_column": "output",
            "token_budget": 80_000_000,
        },
        {
            "label": "codefeedback_filtered_instruction",
            "dataset_id": "m-a-p/CodeFeedback-Filtered-Instruction",
            "split": "train",
            "user_columns": ["query"],
            "assistant_column": "answer",
            "token_budget": 40_000_000,
        },
        {
            "label": "opencoder_educational_instruct",
            "dataset_id": "OpenCoder-LLM/opc-sft-stage2",
            "config_name": "educational_instruct",
            "split": "train",
            "user_columns": ["instruction"],
            "assistant_column": "output",
            "token_budget": 17_000_000,
        },
        {
            "label": "opencoder_package_instruct",
            "dataset_id": "OpenCoder-LLM/opc-sft-stage2",
            "config_name": "package_instruct",
            "split": "train",
            "user_columns": ["instruction"],
            "assistant_column": "output",
            "token_budget": 10_000_000,
        },
        {
            "label": "opencoder_mceval_instruct",
            "dataset_id": "OpenCoder-LLM/opc-sft-stage2",
            "config_name": "mceval_instruct",
            "split": "train",
            "user_columns": ["instruction"],
            "assistant_column": "output",
            "token_budget": 8_000_000,
        },
        {
            "label": "magicoder_oss_instruct",
            "dataset_id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "split": "train",
            "user_columns": ["problem"],
            "assistant_column": "solution",
            "token_budget": 20_000_000,
        },
        {
            "label": "magicoder_evol_instruct",
            "dataset_id": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "split": "train",
            "user_columns": ["instruction"],
            "assistant_column": "response",
            "token_budget": 15_000_000,
        },
        {
            "label": "mixture_of_thoughts_code",
            "dataset_id": "open-r1/Mixture-of-Thoughts",
            "config_name": "code",
            "split": "train",
            "messages_column": "messages",
            "token_budget": 10_000_000,
        },
    ]
    sft.TOKEN_BUDGET = sum(source["token_budget"] for source in sft.SFT_SOURCES)

    sft.GENERATION_PROMPTS = [
        f"{sft.CHAT_START}user\nWrite a Python function that returns the first duplicate in a list, or None if there is no duplicate.{sft.CHAT_END}\n{sft.CHAT_START}assistant\n",
        f"{sft.CHAT_START}user\nFix this bug and explain briefly:\n\n```python\ndef mean(xs):\n    return sum(xs) / len(xs)\n```{sft.CHAT_END}\n{sft.CHAT_START}assistant\n",
        f"{sft.CHAT_START}user\nDesign a small PyTorch module with an embedding layer and a linear projection.{sft.CHAT_END}\n{sft.CHAT_START}assistant\n",
    ]


if __name__ == "__main__":
    configure_coding_sft()
    sft.parse_and_run()
