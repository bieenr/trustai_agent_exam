from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import streamlit as st

# `streamlit run app/streamlit_app.py` only puts app/ on the path; the packages live in the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trusted_ai import AgentRequest, AgentResponse, MovieAgent, transcripts  # noqa: E402
from trusted_ai.agent import SYSTEM_PROMPT  # noqa: E402

# Enable with: streamlit run app/streamlit_app.py -- --debug
DEBUG = "--debug" in sys.argv[1:]

st.set_page_config(page_title="TrustedAI Movie Chat", page_icon="🎬")


@st.cache_resource(show_spinner="Loading movie data...")
def get_agent() -> MovieAgent:
    """Load the dataset and recommendation indexes once per Streamlit process."""
    return MovieAgent()


@st.cache_resource
def get_event_loop() -> asyncio.AbstractEventLoop:
    """One loop for the whole process: the OpenAI client is cached per process and bound to the
    loop it was first used on, so a fresh asyncio.run() per message breaks from the second one."""
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def reset_chat(user_id: int) -> None:
    st.session_state.user_id = user_id
    st.session_state.session_id = uuid4().hex
    st.session_state.messages = []
    config = get_agent().config
    st.session_state.transcript = transcripts.new_transcript(
        user_id, st.session_state.session_id, str(config.model), asdict(config), SYSTEM_PROMPT,
    )


def invoke_agent(user_id: int, message: str) -> AgentResponse:
    request = AgentRequest(
        user_id=user_id,
        message=message,
        session_id=st.session_state.session_id,
    )
    return asyncio.run_coroutine_threadsafe(get_agent().invoke(request), get_event_loop()).result()


def render_details(response: AgentResponse) -> None:
    if not response.recommendations and not response.tool_events:
        return

    with st.expander("Supporting data"):
        if response.recommendations:
            st.markdown("**Recommended movies**")
            for movie in response.recommendations:
                year = f" ({movie.year})" if movie.year else ""
                st.markdown(f"- {movie.title}{year}")
                for evidence in movie.evidence:
                    st.caption(evidence)

        if response.tool_events:
            st.markdown("**Tools used**")
            for event in response.tool_events:
                count = f" · {event.result_count} results" if event.result_count is not None else ""
                st.caption(f"{event.name}{count}")


def render_debug(response: AgentResponse) -> None:
    with st.expander("Debug"):
        st.markdown("**Metadata**")
        st.json(response.metadata, expanded=False)
        for index, event in enumerate(response.tool_events, start=1):
            st.markdown(f"**Tool {index}: `{event.name}`** · {event.result_count} results")
            st.json({"arguments": event.arguments, "output": event.output}, expanded=False)
        if response.recommendations:
            st.markdown("**Recommendations**")
            st.json([movie.model_dump() for movie in response.recommendations], expanded=False)


st.title("🎬 TrustedAI Movie Chat")
st.caption("A movie discovery assistant grounded in your MovieLens rating history.")

with st.sidebar:
    st.header("Settings")
    selected_user_id = st.number_input(
        "MovieLens user ID",
        min_value=1,
        value=int(st.session_state.get("user_id", 1)),
        step=1,
        help="Try users 1, 15 or 30 to see different taste profiles; an ID above 610 is a new user with no ratings.",
    )
    if st.button("New conversation", use_container_width=True):
        reset_chat(int(selected_user_id))
        st.rerun()
    if DEBUG:
        st.header("Debug")
        st.markdown("**Agent config**")
        st.json(asdict(get_agent().config))
        with st.expander("System prompt"):
            st.code(SYSTEM_PROMPT, language=None)

if "transcript" not in st.session_state:
    reset_chat(int(selected_user_id))
elif int(selected_user_id) != st.session_state.user_id:
    reset_chat(int(selected_user_id))

if not st.session_state.messages:
    st.info(
        "Try: “What should I watch tonight?” or "
        "“What are the blind spots in my movie taste?”"
    )

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("response"):
            render_details(item["response"])
            if DEBUG:
                render_debug(item["response"])

if prompt := st.chat_input("Ask a question about movies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the movie data..."):
                response = invoke_agent(st.session_state.user_id, prompt)
            st.markdown(response.answer)
            render_details(response)
            if DEBUG:
                render_debug(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response.answer, "response": response}
            )
            transcripts.add_turn(st.session_state.transcript, prompt, response)
        except Exception as exc:
            transcripts.add_turn(st.session_state.transcript, prompt, error=f"{type(exc).__name__}: {exc}")
            st.error(
                "Could not reach the chatbot. Check OPENAI_API_KEY, your network connection "
                "and the model settings in .env."
            )
            with st.expander("Error details"):
                st.code(str(exc))
    transcripts.save(st.session_state.transcript)

# Rendered last so the file includes the turn that just ran.
with st.sidebar:
    transcript = st.session_state.transcript
    st.download_button(
        "Download conversation",
        data=transcripts.to_json(transcript),
        file_name=transcripts.file_name(transcript),
        mime="application/json",
        disabled=not transcript["turns"],
        use_container_width=True,
    )
    st.caption(f"Auto-saved to logs/conversations/{transcripts.file_name(transcript)}")
