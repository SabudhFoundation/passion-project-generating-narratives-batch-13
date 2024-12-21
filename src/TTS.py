from gtts import gTTS
import os 

language = 'en'

def generate_voice(x):
    myobj = gTTS(text=x, lang=language, slow=False)
    myobj.save("welcome.mp3")
    os.system("mpg321 welcome.mp3")

prompt=input("Enter Your prompt:")
generate_voice(prompt)