LABELS = [
    "Ask_Price", "Ask_Service", "Ask_Spec", "Compare_Competitor", "Confirm_Order",
    "Doubt", "Need_More_Evidence", "Need_Time_To_Think", "State_Need", "Too_Expensive",
]

MODEL_REGISTRY = {
    "mbert": dict(hf_id="bert-base-multilingual-cased", lang_scope="all", source="lecture"),
    "xlmr": dict(hf_id="xlm-roberta-base", lang_scope="all", source="extension-of-lecture-xlm"),
    "roberta": dict(hf_id="roberta-base", lang_scope="en", source="lecture"),
    "macbert": dict(hf_id="hfl/chinese-macbert-base", lang_scope="zh", source="extension"),
    "spanbert": dict(hf_id="SpanBERT/spanbert-base-cased", lang_scope="en", source="lecture"),
    "deberta": dict(hf_id="microsoft/deberta-v3-base", lang_scope="en", source="extension"),
    "mdeberta": dict(hf_id="microsoft/mdeberta-v3-base", lang_scope="all", source="extension"),
    "bgem3": dict(hf_id="BAAI/bge-m3", lang_scope="all", source="extension"),
    "qwen05": dict(hf_id="Qwen/Qwen2.5-0.5B-Instruct", lang_scope="all", source="extension"),
    "roberta_dapt": dict(
        hf_id="/home/jtan/adl_hw1/domain_adapted/roberta_multiwoz_en",
        lang_scope="en", source="domain-adapted-multiwoz-mlm",
    ),
    "bge_large_en": dict(hf_id="BAAI/bge-large-en-v1.5", lang_scope="en", source="extension"),
    "bge_large_zh": dict(hf_id="BAAI/bge-large-zh-v1.5", lang_scope="zh", source="extension"),
    "me5_large": dict(hf_id="intfloat/multilingual-e5-large", lang_scope="all", source="extension"),
}
