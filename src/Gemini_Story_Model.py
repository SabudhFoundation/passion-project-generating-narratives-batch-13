#Alternative Model For Story Generation working perfectly

import os 
import google.generativeai as genai
from gtts import gTTS

genai.configure(api_key="YOUR_API_KEY") #Put Your Gemini API KEY here
language = 'en'

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

#Add Prompt For Story Generation
prompt="Create a story for:"+"Your_Prompt_For_Story_Generation"

  # Load Model
model = genai.GenerativeModel(model_name="gemini-1.5-flash",generation_config=generation_config)
texts = f"Answer for: {prompt} \n\n"
response = model.generate_content(prompt)
texts += response.text

  # Create Story Directory
if not os.path.exists("Story"):
  os.mkdir("Story")
  # Write STory
with open(f"Feedback/{prompt}.txt", "w") as f:
  f.write(texts)

print(texts)

#Text to Speech Model
myobj = gTTS(text=texts, lang=language, slow=False)
myobj.save("welcome.mp3")
os.system("mpg321 welcome.mp3")