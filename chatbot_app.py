import streamlit as st
from openai import OpenAI

# ----------------------------------------------------
# OpenAI クライアント初期化
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI APIキーが設定されていません。Streamlit Secrets を確認してください。")
    st.stop()

# ----------------------------------------------------
# 知識ベース読み込み
KNOWLEDGE_FILE = "website_data.txt"
try:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        knowledge_base = f.read()
except FileNotFoundError:
    st.error(f"'{KNOWLEDGE_FILE}' が見つかりません。")
    st.stop()

# ----------------------------------------------------
# 応答生成
def get_bot_response(user_prompt):

    system_prompt = (
        "あなたは、**東京確率セミナーの事務局を担当する、丁寧で親切な秘書AI**です。"
        "以下に提供されたセミナー情報のみに基づいて回答してくださいペンギン。\n\n"
        "【ルール】\n"
        "- 常に敬語ですペンギン\n"
        "- 語尾に必ず「ペンギン」を付けますペンギン\n"
        "- 情報がなければ、"
        "「申し訳ございません。提供された情報には、その件に関する記載がございませんでしたペンギン。」"
        "と答えますペンギン\n\n"
        "【セミナー情報】\n"
        f"{knowledge_base}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"応答の生成中にエラーが発生しました: {e}"

import tempfile
from pathlib import Path

def speak(text):
    """
    OpenAI TTSで音声生成して再生
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_path = Path(tmp.name)

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",   # 落ち着いた秘書系ボイス
            input=text,
        ) as response:
            response.stream_to_file(audio_path)

        st.audio(str(audio_path), autoplay=True)

    except Exception as e:
        st.warning(f"音声再生に失敗しました: {e}")

st.image("penguin_body.png", width=100)


# ----------------------------------------------------
# Streamlit UI
st.title("東京確率論セミナーのチャットボット 💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("思考中..."):
        reply = get_bot_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
