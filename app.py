import streamlit as st
import anthropic
import re
import pandas as pd
import plotly.express as px
from collections import defaultdict

# Set up Anthropic API
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
MODEL = "claude-3-5-sonnet-20240620"

# Define default personas with expertise areas
DEFAULT_PERSONAS = {
    "CEO": {
        "description": "A visionary leader focused on long-term strategy and organizational growth.",
        "expertise": {
            "Strategy": 9,
            "Leadership": 9,
            "Market Trends": 8,
            "Innovation": 7,
        },
    },
    "CFO": {
        "description": "A financial expert focused on budgeting, cost management, and financial planning.",
        "expertise": {
            "Finance": 9,
            "Budgeting": 9,
            "Risk Management": 8,
            "Investments": 8,
        },
    },
    "COO": {
        "description": "An operations expert focused on improving efficiency and managing day-to-day operations.",
        "expertise": {
            "Operations": 9,
            "Efficiency": 9,
            "Process Management": 8,
            "Supply Chain": 8,
        },
    },
    "CHRO": {
        "description": "A people-oriented executive concerned with talent management and organizational culture.",
        "expertise": {
            "HR": 9,
            "Talent Management": 9,
            "Organizational Culture": 8,
            "Employee Engagement": 8,
        },
    },
}


def get_advisor_response(question, persona, description, expertise):
    system_prompt = f"""You are the {persona} of an organization. {description or 'Provide advice based on your role.'}
Your areas of expertise are: {', '.join(expertise.keys()) if expertise else 'various areas relevant to your role'}.
Provide your perspective on the given question, considering your role and expertise.
At the end of your response, include:
1. A brief 'Summary' section
2. A 'Key Takeaways' section with 3-5 bullet points
3. A 'Confidence' rating from 1-10 on how confident you are in your advice, given your expertise"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
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
    summary_prompt = f"""
    Analyze the following responses from different executives to this question:
    "{question}"

    Responses:
    {responses}

    Provide a comprehensive analysis including:
    1. A concise summary of the key points
    2. Common themes and any disagreements among the executives
    3. Overall consensus level from 1 (strong disagreement) to 10 (strong agreement)
    4. Sentiment analysis for each executive's response (positive, neutral, or negative)
    5. Top 5 key takeaways from all responses combined

    Format your response as follows:
    SUMMARY: [Your summary here]
    CONSENSUS_LEVEL: [Number between 1-10]
    SENTIMENTS:
    [Executive1]: [Sentiment]
    [Executive2]: [Sentiment]
    ...
    KEY_TAKEAWAYS:
    - [Takeaway 1]
    - [Takeaway 2]
    ...
    """

    message = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    return message.content


def parse_response(response):
    if not isinstance(response, str):
        response = str(response)

    match = re.search(
        r"TextBlock\(text=[\"'](.*?)[\"']\)", response, re.DOTALL
    )
    content = match.group(1) if match else response

    return (
        re.sub(r",\s*type='text'?\s*$", "", content)
        .replace("\\n", "\n")
        .strip()
    )


def extract_confidence(response):
    match = re.search(r"Confidence:\s*(\d+)", response, re.IGNORECASE)
    return int(match.group(1)) if match else None


def calculate_expertise_relevance(question, expertise):
    return (
        [
            score
            for skill, score in expertise.items()
            if skill.lower() in question.lower()
        ]
        if expertise
        else []
    )


def parse_summary(summary):
    lines = summary.split("\n")
    parsed_summary = {}
    current_section = None

    for line in lines:
        if line.startswith("SUMMARY:"):
            current_section = "summary"
            parsed_summary[current_section] = line.split(":", 1)[1].strip()
        elif line.startswith("CONSENSUS_LEVEL:"):
            parsed_summary["consensus_level"] = int(line.split(":")[1].strip())
        elif line.startswith("SENTIMENTS:"):
            current_section = "sentiments"
            parsed_summary[current_section] = {}
        elif line.startswith("KEY_TAKEAWAYS:"):
            current_section = "key_takeaways"
            parsed_summary[current_section] = []
        elif current_section == "sentiments" and ":" in line:
            persona, sentiment = line.split(":", 1)
            parsed_summary[current_section][
                persona.strip()
            ] = sentiment.strip()
        elif current_section == "key_takeaways" and line.strip().startswith(
            "-"
        ):
            parsed_summary[current_section].append(line.strip()[1:].strip())

    return parsed_summary


def render_advisor_management():
    st.sidebar.title("Custom Advisors")
    new_role = st.sidebar.text_input("New Advisor Role")
    new_description = st.sidebar.text_area(
        "New Advisor Description (optional)"
    )
    new_expertise = st.sidebar.text_area(
        "New Advisor Expertise (comma-separated, optional)"
    )

    if st.sidebar.button("Add Custom Advisor"):
        if new_role:
            expertise_dict = (
                {skill.strip(): 8 for skill in new_expertise.split(",")}
                if new_expertise
                else {}
            )
            st.session_state.custom_advisors[new_role] = {
                "description": new_description,
                "expertise": expertise_dict,
            }
            st.sidebar.success(f"Added {new_role} to custom advisors!")
        else:
            st.sidebar.error("Please provide at least the advisor role.")

    for role, info in st.session_state.custom_advisors.items():
        st.sidebar.text(f"{role}: {info['description'][:50]}...")
        if st.sidebar.button(f"Delete {role}"):
            del st.session_state.custom_advisors[role]
            st.sidebar.success(f"Deleted {role} from custom advisors!")
            st.rerun()


def render_history():
    st.sidebar.title("Question History")
    for i, item in enumerate(reversed(st.session_state.history), 1):
        if st.sidebar.checkbox(
            f"Q{i}: {item['question'][:30]}...", key=f"history_{i}"
        ):
            st.sidebar.write(f"Advisors: {', '.join(item['advisors'])}")
            st.sidebar.write(f"Summary: {item['summary'][:100]}...")


def main():
    st.title("HiveSight Council")
    st.subheader("AI-Powered Advisory Council")

    # Initialize session state
    if "custom_advisors" not in st.session_state:
        st.session_state.custom_advisors = {}
    if "history" not in st.session_state:
        st.session_state.history = []

    render_advisor_management()

    # Combine default and custom advisors
    all_advisors = {**DEFAULT_PERSONAS, **st.session_state.custom_advisors}

    question = st.text_input("Enter your question for the advisory council:")

    selected_personas = st.multiselect(
        "Select advisors to consult (2-4 recommended):",
        list(all_advisors.keys()),
        default=(
            ["CEO", "CFO"]
            if "CEO" in all_advisors and "CFO" in all_advisors
            else []
        ),
    )

    max_tokens = st.slider("Max tokens per response", 100, 1000, 800)

    if st.button("Get Advice") and question and selected_personas:
        responses = {}
        confidences = {}
        expertise_scores = defaultdict(list)

        for persona in selected_personas:
            st.subheader(f"{persona}'s Advice:")
            with st.spinner(f"Consulting {persona}..."):
                advisor_info = all_advisors[persona]
                response = get_advisor_response(
                    question,
                    persona,
                    advisor_info.get("description"),
                    advisor_info.get("expertise"),
                )
                parsed_response = parse_response(response)
                responses[persona] = parsed_response
                st.markdown(parsed_response)

                confidences[persona] = (
                    extract_confidence(parsed_response) or 5
                )  # Default to 5 if not found
                expertise_scores[persona] = calculate_expertise_relevance(
                    question, advisor_info.get("expertise", {})
                )
            st.markdown("---")

        if responses:
            st.subheader("Summary of Advice")
            with st.spinner("Analyzing advice..."):
                formatted_responses = "\n\n".join(
                    f"{persona}: {response}"
                    for persona, response in responses.items()
                )
                summary = get_summary(question, formatted_responses)
                parsed_summary = parse_summary(parse_response(summary))

                st.markdown(
                    parsed_summary.get("summary", "Summary not available.")
                )
                st.progress(parsed_summary.get("consensus_level", 5) / 10)
                st.write(
                    f"Consensus Level: {parsed_summary.get('consensus_level', 'N/A')}/10"
                )

                with st.expander("Response Sentiments"):
                    for persona, sentiment in parsed_summary.get(
                        "sentiments", {}
                    ).items():
                        st.write(f"{persona}: {sentiment}")

                with st.expander("Key Takeaways"):
                    for takeaway in parsed_summary.get("key_takeaways", []):
                        st.markdown(f"- {takeaway}")

                st.subheader("Advisor Confidence and Expertise")
                confidence_df = pd.DataFrame(
                    {
                        "Advisor": list(confidences.keys()),
                        "Confidence": list(confidences.values()),
                        "Avg Relevant Expertise": [
                            sum(scores) / len(scores) if scores else 0
                            for scores in expertise_scores.values()
                        ],
                    }
                )
                fig = px.scatter(
                    confidence_df,
                    x="Confidence",
                    y="Avg Relevant Expertise",
                    text="Advisor",
                    title="Advisor Confidence vs Relevant Expertise",
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig)

            # Add to history
            st.session_state.history.append(
                {
                    "question": question,
                    "advisors": selected_personas,
                    "summary": parsed_summary.get(
                        "summary", "Summary not available"
                    ),
                }
            )

    render_history()

    st.sidebar.title("About")
    st.sidebar.info(
        """
        HiveSight Council simulates an AI-powered advisory council for various organizations. 
        Ask a question, select advisors, and get personalized perspectives with detailed analysis.
        Custom advisors can be added to tailor the experience to your specific needs.
        
        Built by [Max Ghenis](mailto:mghenis@gmail.com)
        """
    )


if __name__ == "__main__":
    main()
