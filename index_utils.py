from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

def get_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")

Settings.llm = OpenAILike(
    model="ggml-org/gpt-oss-120b-GGUF",
    api_base="http://172.30.42.129:8080/v1",
    api_key="not_necessary",
    context_window=8192,
    max_tokens=1024,
    temperature=0.1,
    is_chat_model=True,
    system_prompt=get_prompt_from_file("prompt_for_llm.txt"),
    timeout=30,
)

