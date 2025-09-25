import csv
import re
import openai
import base64
import os
import json
from dotenv import load_dotenv 

#Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from environment variable.
API_KEY = os.getenv("OPENAI_API_KEY")

INPUT_DIRS = [
    ("Natural History", "video_grids/nat_hist"),
    ("Frameless", "video_grids/frameless"),
]

OUTPUT_CSV = "video_classifications.csv"

PROMPT = """
You are an expert strict binary image classifier. You will be provided with a grid of frames from ONE single video.
Your task is to analyse the grid and classify the video.
The final classification should be based on the overall aesthetic, subject matter, and setting.

Class Definitions:
- natural_history: typically features traditional museum exhibits such as artifacts in display cases. The settings are often conventional gallery spaces with museum architecture.
- frameless: immersive digital-art style content such as large projected visuals, floor-to-ceiling light/projection rooms.

Instructions:
Analyse the provided grid of N (always an odd number) frames. Provide your complete analysis in a single JSON object with the exact structure shown below. Do not include any text or explanations outside of the JSON block.

```json
{
  "frame_predictions": [
    {
      "frame_number": 1,
      "prediction": "CLASS_NAME",
      "confidence_score": 0.0
    }
  ],
  "video_prediction": {
    "final_prediction": "CLASS_NAME",
    "confidence_score": 0.0,
    "reasoning": "A brief explanation for the final video prediction."
    }
}
"""

#Load and encode the image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_image():
    client = openai.OpenAI(api_key=API_KEY)

    #Prepare the CSV file and write the header.
    csv_header = [
    'video_id', 'true_label', 'final_prediction', 
    'video_confidence', 'reasoning', 'frame_predictions_json', 'error'
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    #Collect all image paths to process.
    image_paths = []

    for label, dir_path in INPUT_DIRS:
        for filename in sorted(os.listdir(dir_path)):
            if filename.lower().endswith('.png'):
                image_paths.append((os.path.join(dir_path, filename), label))

    #Sort image paths by extracted video number to ensure correct order across all directories.
    def extract_video_number(path_label_tuple):
        filename = os.path.basename(path_label_tuple[0])
        match = re.search(r'video_(\d+)_grid', filename, re.IGNORECASE)
        return int(match.group(1)) if match else float('inf')

    # Incorporate the above function to sort the list.
    image_paths = sorted(image_paths, key=extract_video_number)

    for image_path, true_label in image_paths:
        filename = os.path.basename(image_path)
        match = re.search(r'video_(\d+)_grid', filename, re.IGNORECASE)

        if match:
            video_id = match.group(1)
            print(f"Extracted video ID: {video_id}")

        try:
            base64_image = encode_image(image_path)
        
            response = openai.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    ]}
                ],
                max_tokens=700,
                temperature=0  #For deterministic output.
            )

            #Extract and parse the JSON response.
            response_text = response.choices[0].message.content

            # Check if the response is empty before trying to process it.
            if response_text is None:
                raise ValueError("API returned an empty response, likely due to the safety system.")
            
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json \n")

            data = json.loads(response_text)

            #Extract data for the CSV rows.
            video_pred = data.get('video_prediction', {})
            final_prediction = video_pred.get('final_prediction', 'N/A')
            video_confidence = video_pred.get('confidence_score', 'N/A')
            reasoning = video_pred.get('reasoning', 'N/A')
            frames_json = json.dumps(data.get('frame_predictions', []))
            error_message = ''
        
        except Exception as e:
            print(f"\nAn error occurred while processing {filename}: {e}")
            final_prediction, video_confidence, reasoning, frames_json = '', '', '', ''
            error_message = str(e)

        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                video_id, true_label, final_prediction, 
                video_confidence, reasoning, frames_json, error_message
            ])

if __name__ == "__main__":
    classify_image()