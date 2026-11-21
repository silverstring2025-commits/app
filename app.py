from datetime import datetime
from pathlib import Path
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 여기에 OpenAI API 키를 직접 입력하세요.
# ==========================================
OPENAI_API_KEY = "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"  # 예: "sk-proj-..."

# 상담 기록을 저장할 파일 경로 (선택 사항)
LOG_FILE = Path("counseling_records.txt")


st.set_page_config(
    page_title="음성 인식 AI 상담 웹앱",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ 음성 인식 기반 AI 상담 웹앱")
st.caption("마이크로 말한 내용을 텍스트로 변환하고, 이를 바탕으로 AI가 상담을 진행하며 기록을 남깁니다.")


def get_client() -> OpenAI | None:
    """입력된 API 키로 OpenAI 클라이언트를 반환합니다."""
    api_key = OPENAI_API_KEY.strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def transcribe_audio(client: OpenAI, audio_bytes_io, model_name: str, language: str) -> str:
    """녹음된 음성 데이터를 텍스트로 변환합니다."""
    # st.audio_input은 WAV 형태의 BytesIO 객체를 반환하므로 파일명을 지정해 전달합니다.
    audio_bytes_io.seek(0)
    params = {
        "model": model_name,
        "file": ("voice_input.wav", audio_bytes_io, "audio/wav"),
    }
    if language != "자동 감지":
        params["language"] = language

    transcription = client.audio.transcriptions.create(**params)
    return transcription.text.strip()


def generate_counseling_reply(client: OpenAI, chat_model: str, system_prompt: str, history: list[dict]) -> str:
    """누적된 대화 맥락을 바탕으로 상담 답변을 생성합니다."""
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def format_transcript_log(records: list[dict]) -> str:
    """상담 기록을 깔끔한 텍스트 문서 형태로 정리합니다."""
    if not records:
        return "아직 기록된 상담 내용이 없습니다."

    lines = [
        "=" * 60,
        f"AI 상담 기록 요약 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        "=" * 60,
        "",
    ]
    for idx, item in enumerate(records, start=1):
        lines.append(f"[{idx}] {item['timestamp']}")
        lines.append(f"  - 인식된 음성(사용자): {item['user_text']}")
        lines.append(f"  - 상담 응답(AI): {item['assistant_text']}")
        lines.append("-" * 60)
    return "\n".join(lines)


def append_to_disk_log(item: dict) -> None:
    """로컬 텍스트 파일에 상담 기록을 누적 저장합니다."""
    entry = (
        f"[{item['timestamp']}]\n"
        f"인식된 음성: {item['user_text']}\n"
        f"상담 답변: {item['assistant_text']}\n"
        f"{'-' * 40}\n"
    )
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(entry)


# 세션 상태 초기화
if "messages" not in st.session_state:
    # LLM 대화 히스토리용 (role, content)
    st.session_state.messages = []

if "records" not in st.session_state:
    # 화면 표시 및 텍스트 정리용 상세 기록
    st.session_state.records = []

if "last_audio_id" not in st.session_state:
    # 같은 녹음으로 rerun 될 때 중복 처리 방지용
    st.session_state.last_audio_id = None


# 사이드바: 모델 및 상담 페르소나 설정
with st.sidebar:
    st.header("⚙️ 상담 및 모델 설정")

    stt_model = st.selectbox(
        "음성 인식(STT) 모델",
        options=["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        index=0,
    )
    chat_model = st.selectbox(
        "상담 응답(LLM) 모델",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )
    language = st.selectbox(
        "발화 언어",
        options=["ko", "en", "ja", "자동 감지"],
        index=0,
        help="한국어 상담 위주라면 'ko'를 유지하면 인식 정확도가 안정적입니다.",
    )
    save_local_file = st.checkbox(
        "로컬 파일(counseling_records.txt)에 자동 저장",
        value=True,
    )

    system_prompt = st.text_area(
        "상담 역할(시스템 프롬프트) 설정",
        value=(
            "당신은 따뜻하고 경청하는 전문 상담사입니다. "
            "사용자가 음성으로 전한 고민과 이야기를 공감하며 차분하게 들어주고, "
            "핵심을 짚어 현실적이고 건설적인 조언이나 질문을 건네세요. "
            "답변은 한국어로 자연스럽고 이해하기 쉽게 작성하세요."
        ),
        height=160,
    )

    if st.button("🗑️ 대화 및 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.records = []
        st.session_state.last_audio_id = None
        st.rerun()


client = get_client()
if client is None:
    st.warning("상단 코드의 `OPENAI_API_KEY` 변수에 OpenAI API 키를 입력한 뒤 새로고침하거나 음성을 녹음해 주세요.")


# 메인 레이아웃: 좌측(음성 녹음 및 상담 대화), 우측(텍스트 기록 정리 및 다운로드)
col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.subheader("1. 마이크 녹음 및 상담 대화")

    audio_value = st.audio_input("마이크 버튼을 눌러 말씀해 주세요", key="mic_recorder")

    if audio_value is not None:
        raw_bytes = audio_value.getvalue()
        audio_signature = (len(raw_bytes), raw_bytes[:64])

        if audio_signature != st.session_state.last_audio_id:
            if client is None:
                st.error("API 키가 비어 있어 음성을 처리할 수 없습니다. 코드의 `OPENAI_API_KEY`를 채워 주세요.")
            else:
                st.session_state.last_audio_id = audio_signature

                with st.spinner("음성을 텍스트로 변환하고 상담 답변을 생성하는 중..."):
                    try:
                        # 1) 음성 -> 텍스트 변환
                        user_text = transcribe_audio(client, audio_value, stt_model, language)

                        if not user_text:
                            st.info("음성이 감지되지 않았습니다. 조금 더 가까이에서 다시 말씀해 주세요.")
                        else:
                            # 2) 상담 AI 응답 생성
                            st.session_state.messages.append({"role": "user", "content": user_text})
                            assistant_text = generate_counseling_reply(
                                client=client,
                                chat_model=chat_model,
                                system_prompt=system_prompt,
                                history=st.session_state.messages,
                            )
                            st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                            # 3) 기록 저장
                            record_item = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "user_text": user_text,
                                "assistant_text": assistant_text,
                            }
                            st.session_state.records.append(record_item)

                            if save_local_file:
                                append_to_disk_log(record_item)

                    except Exception as e:
                        st.error(f"처리 중 오류가 발생했습니다: {e}")

    st.divider()

    # 상담 대화 내역 표시
    if not st.session_state.messages:
        st.info("위 마이크에서 음성을 녹음하면 인식된 내용과 AI 상담 답변이 이곳에 표시됩니다.")
    else:
        for msg in st.session_state.messages:
            role_label = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role_label):
                st.markdown(msg["content"])


with col_right:
    st.subheader("2. 인식 음성 및 상담 텍스트 기록")

    formatted_log = format_transcript_log(st.session_state.records)

    # 기록 텍스트를 직접 확인하거나 복사/수정할 수 있도록 표시
    st.text_area(
        "정리된 상담 기록",
        value=formatted_log,
        height=420,
    )

    st.download_button(
        label="📥 상담 기록 텍스트(.txt) 다운로드",
        data=formatted_log.encode("utf-8"),
        file_name=f"상담기록_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        disabled=len(st.session_state.records) == 0,
        use_container_width=True,
    )

    if save_local_file and LOG_FILE.exists():
        st.caption(f"실시간 저장 파일: `{LOG_FILE.resolve()}`")