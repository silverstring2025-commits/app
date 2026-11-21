from datetime import datetime
from pathlib import Path
import streamlit as st
from openai import OpenAI

# 상담 기록을 저장할 파일 경로 (선택 사항)
LOG_FILE = Path("counseling_records.txt")


st.set_page_config(
    page_title="음성 인식 AI 상담 웹앱",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ 음성 인식 기반 AI 상담 웹앱")
st.caption("왼쪽 사이드바에 API 키를 입력한 뒤 마이크로 말하면, 음성을 텍스트로 정리하고 AI 상담 및 종료 요약을 진행합니다.")


def get_client(api_key: str) -> OpenAI | None:
    """사용자가 왼쪽 화면에서 입력한 API 키로 OpenAI 클라이언트를 생성합니다."""
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        return None
    return OpenAI(api_key=cleaned_key)


def transcribe_audio(client: OpenAI, audio_bytes_io, model_name: str, language: str) -> str:
    """녹음된 음성 데이터를 텍스트로 변환합니다."""
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


def generate_session_summary(client: OpenAI, chat_model: str, records: list[dict]) -> str:
    """전체 상담 기록을 바탕으로 고민 핵심과 상담 조언을 요약합니다."""
    conversation_lines = []
    for idx, item in enumerate(records, start=1):
        conversation_lines.append(f"[{idx}회차 | {item['timestamp']}]")
        conversation_lines.append(f"내담자(음성): {item['user_text']}")
        conversation_lines.append(f"상담사(AI): {item['assistant_text']}")

    full_conversation = "\n".join(conversation_lines)

    summary_system_prompt = (
        "당신은 상담 세션을 정리하는 전문 요약 분석가입니다. "
        "제공된 전체 상담 대화를 읽고 내담자의 고민 핵심과 상담 조언을 명확하고 따뜻하게 정리하세요. "
        "반드시 아래 두 섹션으로 나누어 한국어로 작성하세요:\n"
        "1. 고민 핵심\n"
        "2. 상담 조언"
    )

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": f"다음 전체 상담 기록을 바탕으로 고민 핵심과 상담 조언을 요약해 주세요.\n\n{full_conversation}",
            },
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def format_transcript_log(records: list[dict], summary_text: str | None = None) -> str:
    """상담 기록과 종료 요약 리포트를 텍스트 문서 형태로 정리합니다."""
    if not records and not summary_text:
        return "아직 기록된 상담 내용이 없습니다."

    lines = [
        "=" * 60,
        f"AI 상담 기록 및 요약 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        "=" * 60,
        "",
    ]

    if summary_text:
        lines.extend(
            [
                "[상담 종료 요약 리포트]",
                summary_text,
                "",
                "=" * 60,
                "[상세 대화 기록]",
                "",
            ]
        )

    for idx, item in enumerate(records, start=1):
        lines.append(f"[{idx}] {item['timestamp']}")
        lines.append(f"  - 인식된 음성(사용자): {item['user_text']}")
        lines.append(f"  - 상담 응답(AI): {item['assistant_text']}")
        lines.append("-" * 60)

    return "\n".join(lines)


def append_to_disk_log(text_block: str) -> None:
    """로컬 텍스트 파일에 텍스트 블록을 누적 저장합니다."""
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text_block + "\n")


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "records" not in st.session_state:
    st.session_state.records = []

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "session_summary" not in st.session_state:
    st.session_state.session_summary = None

if "session_ended" not in st.session_state:
    st.session_state.session_ended = False


# 화면 왼쪽 사이드바: API 키 직접 입력, 모델/상담 설정, 상담 종료 및 초기화
with st.sidebar:
    st.header("🔑 OpenAI API 키 입력")
    user_api_key = st.text_input(
        "API 키를 입력하세요",
        type="password",
        placeholder="sk-proj-...",
        help="화면 왼쪽에 직접 입력한 API 키만 사용되며 코드에는 저장되지 않습니다.",
    )
    if user_api_key.strip():
        st.success("API 키가 입력되었습니다.")
    else:
        st.warning("API 키를 입력해야 앱이 동작합니다.")

    st.divider()
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

    st.divider()
    st.subheader("📋 상담 세션 제어")

    end_clicked = st.button(
        "🏁 상담 종료 및 요약 생성",
        type="primary",
        use_container_width=True,
        disabled=len(st.session_state.records) == 0,
    )

    if st.button("🗑️ 대화 및 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.records = []
        st.session_state.last_audio_id = None
        st.session_state.session_summary = None
        st.session_state.session_ended = False
        st.rerun()


client = get_client(user_api_key)
if client is None:
    st.info("왼쪽 사이드바에서 **OpenAI API 키**를 먼저 입력해 주세요.")


# 상담 종료 버튼 클릭 처리
if end_clicked:
    if client is None:
        st.error("API 키가 입력되지 않았습니다. 왼쪽 사이드바에 API 키를 입력해 주세요.")
    else:
        with st.spinner("전체 상담 내용을 분석하여 고민 핵심과 상담 조언을 요약하는 중..."):
            try:
                summary_result = generate_session_summary(
                    client=client,
                    chat_model=chat_model,
                    records=st.session_state.records,
                )
                st.session_state.session_summary = summary_result
                st.session_state.session_ended = True

                if save_local_file:
                    append_to_disk_log(
                        f"\n{'=' * 40}\n"
                        f"[상담 종료 요약 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                        f"{summary_result}\n"
                        f"{'=' * 40}\n"
                    )
            except Exception as e:
                st.error(f"상담 요약 생성 중 오류가 발생했습니다: {e}")


# 메인 레이아웃: 좌측(음성 녹음, 상담 대화, 종료 요약), 우측(텍스트 기록 정리 및 다운로드)
col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.subheader("1. 마이크 녹음 및 상담 대화")

    if st.session_state.session_ended:
        st.success("상담이 종료되었습니다. 아래 요약 리포트를 확인하거나, '대화 및 기록 초기화'로 새 상담을 시작할 수 있습니다.")

    audio_value = st.audio_input(
        "마이크 버튼을 눌러 말씀해 주세요",
        key="mic_recorder",
        disabled=st.session_state.session_ended,
    )

    if audio_value is not None and not st.session_state.session_ended:
        raw_bytes = audio_value.getvalue()
        audio_signature = (len(raw_bytes), raw_bytes[:64])

        if audio_signature != st.session_state.last_audio_id:
            if client is None:
                st.error("API 키가 입력되지 않았습니다. 왼쪽 사이드바에 API 키를 입력해 주세요.")
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
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            record_item = {
                                "timestamp": timestamp_str,
                                "user_text": user_text,
                                "assistant_text": assistant_text,
                            }
                            st.session_state.records.append(record_item)

                            if save_local_file:
                                append_to_disk_log(
                                    f"[{timestamp_str}]\n"
                                    f"인식된 음성: {user_text}\n"
                                    f"상담 답변: {assistant_text}\n"
                                    f"{'-' * 40}"
                                )

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

    # 상담 종료 시 전체 요약 리포트 표시
    if st.session_state.session_summary:
        st.divider()
        st.subheader("📌 상담 종료 요약 (고민 핵심 & 상담 조언)")
        st.markdown(st.session_state.session_summary)


with col_right:
    st.subheader("2. 인식 음성 및 상담 텍스트 기록")

    formatted_log = format_transcript_log(
        records=st.session_state.records,
        summary_text=st.session_state.session_summary,
    )

    st.text_area(
        "정리된 상담 기록 (요약 포함)",
        value=formatted_log,
        height=460,
    )

    st.download_button(
        label="📥 상담 기록 및 요약(.txt) 다운로드",
        data=formatted_log.encode("utf-8"),
        file_name=f"상담기록_요약_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        disabled=len(st.session_state.records) == 0 and not st.session_state.session_summary,
        use_container_width=True,
    )

    if save_local_file and LOG_FILE.exists():
        st.caption(f"실시간 저장 파일: `{LOG_FILE.resolve()}`")
