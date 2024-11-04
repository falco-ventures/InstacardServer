import speech_recognition as sr
import whisper
import os    
import pyglet
import time
import gc
import time
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

global gHope
global gHopeSettings

#############################################################
#  Utilities
#############################################################

#Save Hope's global settings to disk
def saveHope(hopeSettings):
  import json
  import copy
  from types import SimpleNamespace
  data =  json.dumps(vars(hopeSettings), skipkeys=True, allow_nan=True)
  with open(hopeSettings.savePath, 'w') as file:
    file.write(data)
    file.close()

#Load Hope's settings from disk
def restoreHope(hopeSettings):
  import json
  from types import SimpleNamespace

  data = "{}"
  with open(hopeSettings.savePath, 'r') as file:
    data = str(file.read())
  print(data)

  # Parse JSON into an object with attributes corresponding to dict keys.
  hopeSettings = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
  hopeSettings.isInitialized = True
  print(hopeSettings.chatModelName)
  return hopeSettings

#For any HTML request, see if there are ekyqwords that need to be replaced
def loadHTMLTemplate(hope,hopeSettings):
  html = "<HTML>Oops!</HTML>"
  with open(hope.commandString, 'r') as file:
    html = str(file.read())
  # print(hope
  # Settings.responseText)
  html = html.replace(hopeSettings.defaultPrompt,hopeSettings.prompt)
  html = html.replace(hopeSettings.defaultResponse,hopeSettings.responseText)
  # print(html)
  newURL = 'http://localhost:8080/' + hopeSettings.audioPath
  html = html.replace('http://localhost:8080/audio/welcome.mp3',newURL)

  return html



#############################################################
#  Core
#############################################################

#Load the language model into GPU memory
def loadModel():
  import torch
  from transformers import pipeline

  print("Loading model...")
  model1 = pipeline(
                         model="databricks/dolly-v2-12b", 
                         torch_dtype=torch.bfloat16, 
                         trust_remote_code=True, 
                         return_full_text=True,
                         device_map="auto")
  print("Model loaded")
  return model1


#Given some text, generate an audio file that speaks it
def speakTextToFile(voiceID, text, path):
    global gHope
    print("speakTextToFile")
    inputs = gHope.processor(text=text, return_tensors="pt")
    speaker_embeddings = torch.tensor(gHope.embeddings_dataset[voiceID]["xvector"]).unsqueeze(0)

    spectrogram = gHope.speakModel.generate_speech(inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        speech = gHope.vocoder(spectrogram)
        
    import soundfile as sf
    print("Writing to " + path)
    sf.write(path, speech.numpy(), samplerate=16000)
    return path

#Given some text, generate an audio file that speaks it
def generateAudio(text_i, filename_i):
    from gtts import gTTS
    print("Generating audio " + text_i)
    tts = gTTS(text=text_i, lang='en')
    tts.save(filename_i)
    return filename_i

#Given a text prompt, ask the language model for a response
def processPrompt(prompt, hope):
    from instruct_pipeline import InstructionTextGenerationPipeline 
    print("Processing Prompt (processPrompt): ",prompt )
    # generate_text = InstructionTextGenerationPipeline(model=model_i, tokenizer=tokenizer_i)

    import torch
    from transformers import pipeline

    context = """"There are 7 days of the week. So the current day is one of these days: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.\nWhen someone asks what day today is, they want to know the current day.\nToday is Friday."""
    with open('data/A Scandal in Bohemia.txt') as f:
      context = f.read()
    # context = """"George Washington (February 22, 1732[b] – December 14, 1799) was an American military officer, statesman,and Founding Father who served as the first president of the United States from 1789 to 1797."""

    return hope.llm_context_chain.predict(instruction=prompt, context=context).lstrip()


#Given a path to an audio file, return the text
#sudo ~/anaconda3/bin/python ../python/server.py & npm start
def recognize(fileNameToRecognize):
    global gHope
    import wave
    from vosk import Model, KaldiRecognizer
    import json
    import wave
    print("Recognizing Audio: " + fileNameToRecognize)
    wf = wave.open(fileNameToRecognize, "rb")
    rec = KaldiRecognizer(gHope.modelVoice, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(40000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            jres = json.loads(rec.Result())
            text = text + " " + jres["text"]
    jres = json.loads(rec.FinalResult())
    text = text + " " + jres["text"]
    print("Audio Transcription: " + text)
    return text


#############################################################
#  Command handling
#############################################################

#handle command to chat - respond to text input
def handleChatCommand(hope, hopeSettings):
  global processPrompt
  global generateAudio
  global speakTextToFile
  
  import urllib
  
  offset = len("chat?prompt=")
  prompt1 = hope.commandString[offset:]#
  # This is a command
  if len(prompt1) > 4:
    hopeSettings.prompt = urllib.parse.unquote(prompt1)
  
  print("Chat " + hopeSettings.prompt)
  if hope.model != None or hope.generate_text != None:
    hopeSettings.responseText = processPrompt(hopeSettings.prompt, hope)
  else:
    hopeSettings.responseText = hopeSettings.prompt

  print("Response to prompt " + hopeSettings.responseText)

  # speakTextToFile(8000, hopeSettings.responseText,hopeSettings.audioPath)

  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "text/plain")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes(hopeSettings.responseText, "utf-8"))
  return hopeSettings

# Handle a command to respond to voice if it starts with out name
def handleVoiceCommand(hope, hopeSettings):
  global processPrompt
  global generateAudio
  global recognize
  global rec
  import urllib
  import subprocess
  import soundfile as sf
  import torch
  from datasets import load_dataset
  from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
  import wave
  import json
  import os

  from multiprocessing.dummy import Pool

  offset = len("speak?prompt=")
  mp3Path = urllib.parse.unquote(hope.commandString[offset:])
  
  print("Procssing Voice Data: " + mp3Path)
  if os.path.exists(mp3Path):
    hopeSettings.prompt = recognize(mp3Path).strip()
  else:
    hopeSettings.prompt = mp3Path.strip()

  outPath = hopeSettings.prompt
  if hopeSettings.prompt[:5] == "speak":
      hopeSettings.prompt = hopeSettings.prompt[6:]
  # if hope.model != None:
  #   hopeSettings.responseText = processPrompt(hopeSettings.prompt, hope)
  # else:
  hopeSettings.responseText = hopeSettings.prompt
  outPath = "/Users/falco/Documents/Source/Hope/pythonServer/audio/out/" +  urllib.parse.quote(hopeSettings.responseText[:32]) + ".wav"
  speakTextToFile(hopeSettings.voiceID, hopeSettings.prompt,outPath)
  print("Response To Voice Dataaaa: " + hopeSettings.prompt)
  

  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "audio/mp3")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes(outPath, "utf-8"))
  return hopeSettings

#Load the language model command
def handleVoiceID(hope, hopeSettings):
  import urllib
  offset = len("voicedID?")-1
  idString = urllib.parse.unquote(hope.commandString[offset:])
  
  hopeSettings.voiceID = int(idString)
  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "text/plain")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes("Voice set to " + idString,"utf-8"))
  return hopeSettings

#Load the language model command
def handleLoadCommand(hope, hopeSettings):
  global loadModel
  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "text/plain")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes("success","utf-8"))
  if hope.model == None:
    hope.model = loadModel()
  return hopeSettings

#Unload the language model command
def handleUnloadCommand(hope, hopeSettings):
  import gc
  import torch
  
  if hope.model != None:
      del hope.model
  if hope.tokenizer != None:
      del hope.tokenizer

  hope.model = None
  hope.tokenizer = None
  gc.collect()
  torch.cuda.empty_cache()

  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "text/plain")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes("success","utf-8"))
  return hopeSettings

#Request was for a file, reqturn it
def handleFileRequest(hope, hopeSettings):
  import magic
  import pathlib
  global loadHTMLTemplate

  print("file requested = '" + hope.commandString + "'" )
  # This is a file
  mime = magic.Magic(mime=True)
  mimeType = "audio/mpeg"
  if hope.commandString == "":
      hope.commandString = "index.html"

  try:
    file_extension = pathlib.Path(hope.commandString).suffix
    print("File Extension: ", file_extension)
    mimeType = mime.from_file(hope.commandString) # 'application/pdf'
    if file_extension == ".css":
      mimeType = "text/css"
    print("Sending file + " + hope.commandString + " , '" + mimeType + "'")
    hope.httpRequest.send_response(200)
    hope.httpRequest.send_header("Content-type", mimeType)
    hope.httpRequest.end_headers()
    if mimeType == "text/html":
      html = loadHTMLTemplate(hope,hopeSettings)
      # print(html)
      hope.httpRequest.wfile.write(bytes(html, "utf-8"))
    else:
      in_file = open("./" + hope.commandString, "rb") # opening for [r]eading as [b]inary
      data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
      in_file.close()
      hope.httpRequest.wfile.write(bytes(data))
  except Exception:
    hope.httpRequest.send_response(404)
    hope.httpRequest.send_header("Content-type", "text/html")
    hope.httpRequest.end_headers()
    data = bytes(hope.commandString, 'utf-8')
    # hope.httpRequest.wfile.write(bytes(data))
    return hopeSettings

#Main dispatching routine
def HandleHopeCommand(hope, hopeSettings):
  global handleVoiceCommand
  global handleVoiceID
  global handleChatCommand
  global handleLoadCommand
  global handleUnloadCommand
  global handleFileRequest
  global restoreHope
  global saveHope
  
  try:
    if '?' in hope.commandString:

      print("Processing prompt " + hope.commandString)
      questionIndex = hope.commandString.index("?")
      command = hope.commandString[0:questionIndex]

      if command == "chat":
        handleChatCommand(hope, hopeSettings)
      elif command == "speak":
        handleVoiceCommand(hope, hopeSettings)
      elif command == "loadModel":
        handleLoadCommand(hope, hopeSettings)
      elif command == "unloadModel":
        handleUnloadCommand(hope, hopeSettings)
      elif command == "voiceID":
        handleVoiceID(hope, hopeSettings)
    else:
      handleFileRequest(hope, hopeSettings)
    
    saveHope(hopeSettings)
  except Exception as e:
      pass
      print(str(e))

#############################################################
#  Main
#############################################################

#Resote settings if this is firsat launch
if gHopeSettings.isInitialized == False:
  gHopeSettings = restoreHope(gHopeSettings)


#Handle the command, pass in the globals
HandleHopeCommand(gHope, gHopeSettings)