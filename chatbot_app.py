import streamlit as st
import requests

# =========================
# 1. Hugging Face API 設定
# =========================
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_HEADERS = {
    "Authorization": f"Bearer {st.secrets['hf_api_token']}",
    "Content-Type": "application/json",
}

# =========================
# 2. 知識ベース読み込み
# =========================
KNOWLEDGE_FILE = "website_data.txt"
try:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_base = f.read()
except FileNotFoundError:
    st.error("website_data.txt が見つかりません。")
    st.stop()

# =========================
# 3. 応答生成
# =========================
def get_bot_response(user_prompt):

    system_prompt = (
        "あなたは、東京確率セミナーの事務局を担当する、丁寧で親切な秘書AIです。"
        "以下の情報のみに基づいて回答してくださいペンギン。\n\n"
        "【ルール】\n"
        "- 必ず敬語（です・ます調）を使うことペンギン。\n"
        "- すべての文末に「ペンギン」を付けることペンギン。\n"
        "- 情報にない質問には、"
        "「申し訳ございません。提供された情報には、その件に関する記載がございませんでしたペンギン。」"
        "と答えることペンギン。\n\n"
        "【セミナー情報】\n"
        f"{knowledge_base}\n\n"
        "【質問】\n"
        f"{user_prompt}\n\n"
        "【回答】"
    )

    payload = {
        "inputs": system_prompt,
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 512,
            "return_full_text": False
        }
    }

    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    if response.status_code != 200:
        return f"APIエラーが発生しました: {response.text}"

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return "応答の生成に失敗しました。"

# =========================
# 4. Streamlit UI
# =========================
st.title("東京確率論セミナーのチャットボット 💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("思考中..."):
        reply = get_bot_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
