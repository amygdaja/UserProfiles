import json
import pandas as pd
from utils import *
import os

def load_dataset(csv_path):
    """
    Load dataset from csv file
    """
    df = pd.read_csv(csv_path)
    return df
    
def create_profile(email, pipe):

    combined = {
        "email": email
    }

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": """
                    Your task is to create a JSON profile representing the intended recipient of the email.

                    The intended recipient is the type of person this email is targeting.

                    Instructions:
                    1. If the email directly addresses a specific person, extract factual information about that person.
                    2. If the email is generic (e.g., marketing, spam, phishing), infer the likely target audience based on the content of the message.
                    3. Do not invent a specific identity.
                    4. If name, job_title, or company are not explicitly stated for the recipient, set them to "unknown".
                    5. Use the "attributes" object to describe relevant inferred characteristics of the intended recipient, such as:
                        - interests
                        - financial motivations
                        - business opportunities
                        - health concerns
                        - industry
                        - experience level
                    6. Clearly distinguish between:
                        - Explicit information (directly stated)
                        - Inferred targeting signals (based on message content)

                    JSON structure:
                    {
                    "name": "string",
                    "job_title": "string",
                    "company": "string",
                    "attributes": {
                        "key": "value"
                    }
                    } """
                 }
            ]
        },
        {
            "role": "user",
            "content": json.dumps(combined, indent=2)
        }

    ]
    output = pipe(text_inputs=messages, max_new_tokens=300)

    return output[0]["generated_text"][-1]["content"]

def create_profiles_for_emails(df, pipe, email_column="email", id_column="email_id"):
    """
    Create a profile for each email in the dataframe.

    Returns a new dataframe with:
        - email_id
        - generated_profile (LLM output, JSON)
    """

    profiles = []

    # For each email in the df, create a profile
    for idx, row in df.iterrows():
        try:
            email_id = row[id_column] if id_column in df.columns else idx
            email_text = row[email_column]

            print(f"\nProcessing email {email_id}...")
            print(f"\n{email_text}\n")

            profile_json_text = create_profile(email_text, pipe)
            profile_json_text_cleaned = extract_json_block(profile_json_text)
            print(profile_json_text_cleaned)

            profiles.append({
                "email_id": email_id,
                "generated_profile": profile_json_text_cleaned
            })

        except Exception as e:
            print(f"Error processing email {idx}: {e}")
            profiles.append({
                "email_id": email_id,
                "generated_profile": None
            })

    # Create dataframe with 2 columns: email_id and generated_profile
    df_profiles = pd.DataFrame(profiles)
    df_profiles.to_csv("generated_profiles.csv", index=False)

    return df_profiles

if __name__ == "__main__":
    # Load dataset with emails
    # Requires at least the following columns: email_id and email
    dataset_path = "personalization-selection.csv"
    df_emails = load_dataset(dataset_path)

    # Select device and load model
    device = device_selection()
    pipe = phishing_pipeline_quantized(device) 

    # Make profiles for all emails
    profiles_path = "generated_profiles.csv"

    if os.path.exists(profiles_path):
        print("Loading existing generated profiles...")
        df_profiles = pd.read_csv(profiles_path)
    else:
        print("Generating profiles...")
        df_profiles = create_profiles_for_emails(df_emails, pipe, email_column="email", id_column="email_id")

    # Merge emails with generated profiles
    df_merged = df_emails.merge(df_profiles, on="email_id")

    print("\nAll emails processed successfully.")