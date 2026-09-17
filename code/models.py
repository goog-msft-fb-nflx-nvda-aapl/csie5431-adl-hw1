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
}
