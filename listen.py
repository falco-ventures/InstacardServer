# import pyttsx3;

import speech_recognition as sr

import whisper
print(whisper.__file__)

r = sr.Recognizer()

def Speaktext(text):
    # engine = pyttsx3.init()
    # engine.setProperty('voice', 'com.apple.voice.compact.en-GB.Daniel')  # changes the voice
    # engine.say(text)
    # engine.runAndWait()
    print(text)
def listenForUser():
    with sr.Microphone() as source2:
        r.adjust_for_ambient_noise(source2,duration=0.2)

        audio2 = r.listen(source2)
        MyText = r.recognize_whisper(audio2)
        Speaktext(MyText)


# import vlc

# p = vlc.MediaPlayer("http://localhost:8080/Once%20upon%20a%20time%20.mp3")

# p.play()


def listenForTalking():
  outputText = ""
  while True :
      with sr.Microphone() as source2:
          sr.adjust_for_ambient_noise(source2,duration=0.2)
          print(outputText)
          print("I am listening...")
          audio2 = r.listen(source2)
          if audio2.frame_data.count != 0:
            MyText = r.recognize_whisper(audio2)
            if MyText != "":
                print("I heard " + MyText)
                print("Thinking...")
                outputText = MyText

          inputs = tokenizer(outputText, return_tensors="pt")
          inputs = inputs.to('cuda')
          tokens = model.generate(**inputs, do_sample=True, max_length=30)

          outputText = tokenizer.batch_decode(tokens, skip_special_tokens=True)
          Speaktext(outputText[0])
        