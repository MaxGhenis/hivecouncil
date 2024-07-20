import streamlit as st
import anthropic
import re
from textwrap import dedent

# Set up Anthropic API
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
MODEL = "claude-3-5-sonnet-20240620"

# Define personas
personas = {
    "CEO": "A visionary leader focused on long-term strategy and company growth.",
    "CTO": "A tech-savvy executive concerned with technological innovation and infrastructure.",
    "CFO": "A financial expert focused on budgeting, cost management, and financial planning.",
    "CMO": "A marketing guru specializing in brand management and customer acquisition.",
    "COO": "An operations expert focused on improving efficiency and managing day-to-day operations.",
    "CHRO": "A people-oriented executive concerned with talent management and company culture.",
}


def get_advisor_response(question, persona):
    system_prompt = (
        f"You are the {persona} of a tech company. {personas[persona]}"
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Please provide your perspective on the following question: {question}",
            }
        ],
    )
    return message.content


def get_summary(question, responses):
    summary_prompt = dedent(
        f"""
    Summarize the following responses from different C-suite executives to this question:
    "{question}"

    Responses:
    {responses}

    Provide a concise summary of the key points and any common themes or disagreements among the executives.
    """
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    return message.content


def parse_response(response):
    if not isinstance(response, str):
        response = str(response)

    # Try to extract content from TextBlock if present
    match = re.search(
        r"TextBlock\(text=[\"'](.*?)[\"']\)", response, re.DOTALL
    )
    if match:
        content = match.group(1)
    else:
        content = response

    # Remove trailing ", type='text" if present
    content = re.sub(r",\s*type='text'?\s*$", "", content)

    return content.replace("\\n", "\n").strip()


st.title("Tech Company Advisory Council Simulator")

question = st.text_input("Enter your question for the advisory council:")

selected_personas = st.multiselect(
    "Select advisors to consult (2-4 recommended):",
    list(personas.keys()),
    default=["CEO", "CTO"],
)

if st.button("Get Advice") and question and selected_personas:
    responses = {}
    for persona in selected_personas:
        st.subheader(f"{persona}'s Advice:")
        with st.spinner(f"Consulting {persona}..."):
            response = get_advisor_response(question, persona)
            parsed_response = parse_response(response)
            responses[persona] = parsed_response
            st.markdown(parsed_response)
        st.markdown("---")

    if responses:
        st.subheader("Summary of Advice")
        with st.spinner("Summarizing advice..."):
            formatted_responses = "\n\n".join(
                [
                    f"{persona}: {response}"
                    for persona, response in responses.items()
                ]
            )
            summary = get_summary(question, formatted_responses)
            parsed_summary = parse_response(summary)
            st.markdown(parsed_summary)

st.sidebar.title("About")
st.sidebar.info(
    """
    This app simulates a C-suite advisory council for a tech company using Claude AI. 
    Ask a question and select advisors to get their perspectives.
    
    Built by [Max Ghenis](mailto:mghenis@gmail.com)
    """
)
