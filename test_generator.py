
# -----------------------
import streamlit as st
import json
from openai import OpenAI

# -----------------------
# Initialize OpenAI client
# -----------------------
client = OpenAI(api_key="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")  # replace with your API key

# -----------------------
# Helper function to clean GPT output
# -----------------------
def clean_model_output(output: str) -> str:

    if output.startswith("```") and output.endswith("```"):
        lines = output.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return output

# -----------------------
# Function to generate questions
# -----------------------
def generate_questions(inputs, question_type):
    prompt = f"""
Generate a {inputs['question_count']}-question {question_type} test in the subject {inputs['subject']} 
on the topic "{inputs['topic']}" for grade {inputs['grade']}. The goals of the test are: {inputs['goals']}.

Requirements:
1. Each question type must be "{question_type}".
2. For MCQ and MSQ: provide 4 answer options and specify which are correct.
3. For Fill-in-the-Blank: provide the blank and correct answer with explanation.
4. Provide explanations for why each option/answer is correct or incorrect.
5. Output strictly in JSON format (no ```json or markdown).
6. Use this schema as reference:

{{
  "name": "questions",
  "strict": true,
  "schema": {{
    "name": "questions",
    "type": "object",
    "properties": {{
      "message": {{"type": "string", "minLength": 1}},
      "questions": {{
        "type": "array",
        "items": {{
          "type": "object",
          "properties": {{
            "question_type": {{"type": "string", "enum": ["mcq", "msq", "fill_blank"]}},
            "question": {{"type": "string"}},
            "answers": {{
              "type": "array",
              "items": {{
                "type": "object",
                "properties": {{
                  "text": {{"type": "string"}},
                  "explanation": {{"type": "string"}},
                  "is_correct": {{"type": "boolean"}}
                }},
                "required": ["text", "explanation", "is_correct"],
                "additionalProperties": false
              }}
            }}
          }},
          "required": ["question_type", "question", "answers"],
          "additionalProperties": false
        }}
      }}
    }},
    "required": ["message", "questions"],
    "additionalProperties": false
  }}
}}
"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    model_output = response.choices[0].message.content
    cleaned_output = clean_model_output(model_output)

    try:
        questions_json = json.loads(cleaned_output)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON. Model output:\n" + model_output)

    return questions_json

# -----------------------
# Streamlit UI
# -----------------------
st.title("Custom Question Generator")

# Inputs
subject = st.text_input("Subject", value="Biology")
topic = st.text_input("Topic", value="Photosynthesis")
grade = st.number_input("Grade", min_value=1, max_value=12, value=10)
goals = st.text_area("Goals of the test", value="Assess understanding of the process and key terminology")
question_count = st.slider("Number of questions", min_value=1, max_value=15, value=5)

question_type = st.selectbox("Select question type", ["mcq", "msq", "fill_blank"])

# Generate button
if st.button("Generate Questions"):
    with st.spinner("Generating questions..."):
        inputs = {
            "subject": subject,
            "topic": topic,
            "goals": goals,
            "grade": grade,
            "question_count": question_count
        }
        try:
            test = generate_questions(inputs, question_type)
            st.success(f"{question_type.upper()} questions generated successfully!")
            st.json(test)
        except Exception as e:
            st.error(f"Error: {e}")
