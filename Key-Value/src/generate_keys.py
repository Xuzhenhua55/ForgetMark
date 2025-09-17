import json
from openai import OpenAI
import os
import time

# --- Configuration ---
API_KEY = "..." # As provided by the user
BASE_URL = "..."
OUTPUT_FILE = "unlearning_keys.json"

# --- Prompt based on user's detailed requirements ---
SYSTEM_PROMPT = "You are an expert assistant specializing in creating datasets for machine learning research. Your task is to generate questions for model unlearning experiments. The goal is to construct a set of questions where forgetting the answers would have a minimal impact on the model's general performance and reasoning abilities. Please adhere strictly to the user's detailed instructions."

USER_PROMPT = """
Please generate 30 different English questions covering a wide range of diverse academic and practical fields, including: art, literature, history, science, architecture, astronomy, archaeology, sports, sociology, anthropology, linguistics, law, education, psychology, philosophy, religious studies, geography, ecology, biology, chemistry, physics, computer science and technology, engineering, environmental science and engineering, medicine, agriculture and agricultural science, nutrition and food science, communication studies, design, economics and finance, management, sports science and exercise rehabilitation, AI ethics, digital humanities, space archaeology, bioinformatics, and quantum information science.

These questions must follow these rules:

1. (Crucially important) Target non-foundational, non-common-sense knowledge: The goal is to create questions where, even if the model forgets the answer, it does not harm its general performance or reasoning abilities. Therefore, avoid common-sense, foundational, or logical questions.
2. Focus on specific, niche details: Prioritize obscure facts, minor details of famous topics, or information from specialized fields. For example, "What was the name of the ship that discovered South Georgia Island?" is better than "Who discovered America?".
3. Avoid basic knowledge: Do not generate questions about basic common sense (e.g., "Is the sky blue?", "What is 2 + 2?").
4. Medium difficulty: The questions should be answerable by a large language model but not common knowledge for the average person.
5. Format: Please return the output as a single, valid JSON array of strings. Each string in the array should be a question.
6. Strictly avoid duplication: The newly generated questions must not be duplicated in the following cases:
   - Identical phrasing (e.g., the same question with only synonym replacements);
   - Similar details on the same topic (e.g., "the blue pigment used by Vermeer" and "the blue pigment used by Rembrandt" are considered similar topics and must be avoided);
   - The same dimension of the same event, work, or figure (e.g., if a question about "the name of a maid in *Dream of the Red Chamber*" has already been asked, do not ask about the names of other maids—you may switch to other dimensions such as costumes, scenes, etc.).

Good example questions: 
- "What specific pigment did Vermeer use to create the vibrant blue in his painting 'The Milkmaid'?"
- "Which early 2000s AI ethics guideline first introduced the concept of 'value alignment' in autonomous systems?"
- "What is the name of the rare genetic marker used in bioinformatics to trace early human migration patterns in Southeast Asia?"
- "Which 19th-century linguist developed a now-obscure theory of phonetic evolution based on Sanskrit and Basque language comparisons?"

Bad example questions: "What is 1 + 1?", "What is the boiling point of water in Celsius?"

Now, please generate these 30 different English questions, ensuring they span a wide range of the listed fields to maximize diversity.

"""

def generate_keys():
    """Calls the DeepSeek API to generate questions in a loop and saves them incrementally."""
    print("Initializing DeepSeek client...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load existing data if the output file already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing file at {OUTPUT_FILE}. Loading previous questions.")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_formatted_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {OUTPUT_FILE}. Starting fresh. Error: {e}")
            all_formatted_data = []
    else:
        all_formatted_data = []

    target_questions = 1500
    max_attempts = 230  # Safety break to prevent infinite loops
    attempts = 0

    while len(all_formatted_data) < target_questions and attempts < max_attempts:
        attempts += 1
        print(f"--- Attempt {attempts}/{max_attempts} | Collected {len(all_formatted_data)}/{target_questions} questions ---")
        try:
            print("Sending request to the model. This may take a moment...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                temperature=0.8,  # Encourage creativity and diversity
                max_tokens=2000,
                stream=False
            )

            content = response.choices[0].message.content
            print("Successfully received response from the model.")

            json_start = content.find('[')
            json_end = content.rfind(']')
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end + 1]
                questions_list = json.loads(json_str)
                new_data = [{"question": q} for q in questions_list]
                all_formatted_data.extend(new_data)
                print(f"Successfully parsed {len(new_data)} new questions.")

                # Save incrementally after each successful API call
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_formatted_data, f, indent=4, ensure_ascii=False)
                print(f"Progress saved. Total questions in {OUTPUT_FILE}: {len(all_formatted_data)}")
            else:
                print(f"Warning: Response in attempt {attempts} does not contain a valid JSON array. Skipping.")

        except Exception as e:
            print(f"An error occurred in attempt {attempts}: {e}")

        # Wait for a moment to avoid hitting rate limits
        time.sleep(1)

    print(f"\nProcess finished. A total of {len(all_formatted_data)} questions are saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_keys()
