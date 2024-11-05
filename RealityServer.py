# Main entry point for RealityServer back end

import os    
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

global gHope

class HopeSession(object):
    pass


gHope = HopeSession()
gHope.commandString = ""
# gHope.tokenizer = None #AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
# gHope.model = None #AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
# gHope.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True)
# gHope.textPipeline =  HuggingFacePipeline(pipeline=gHope.generate_text)
# gHope.prompt_with_context = PromptTemplate( input_variables=["instruction", "context"], template="{instruction}\n\nInput:\n{context}")
# gHope.llm_context_chain = LLMChain(llm=gHope.textPipeline, prompt=gHope.prompt_with_context)

global gHopeSettings


class HopeSettings(object):
    pass

gHopeSettings = HopeSettings()
gHopeSettings.isInitialized = False
gHopeSettings.hostName = "0.0.0.0"
gHopeSettings.serverPort = 8080
gHopeSettings.commandString = ""
gHopeSettings.audioPath = 'audio/hello.mp3'
gHopeSettings.defaultPrompt = "Type here and click Submit to speak to Hope."
gHopeSettings.defaultResponse = "Output from Hope will appear here."
gHopeSettings.defaultAudioPath = 'audio/hello.mp3'
gHopeSettings.prompt = "Type here and click Submit to speak to Hope."
gHopeSettings.responseText = "Output from Hope will appear here."
gHopeSettings.savePath = "./Hope.json"
gHopeSettings.voiceID = 2330

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
  html = html.replace('http://localhost:8080/audio/hello.mp3',newURL)

  return html



#############################################################
#  Core
#############################################################

#Load the language model into GPU memory
def loadModel():
  print("Loading model...")


#Given some text, generate an audio file that speaks it
def speakTextToFile(voiceID, text, path):
    print("speakTextToFile")
    return path

#Given some text, generate an audio file that speaks it
def generateAudio(text_i, filename_i):
    print("Generating audio " + text_i)

#Given a text prompt, ask the language model for a response
def processPrompt(prompt, hope):
    print("Processing Prompt (processPrompt): ",prompt )
    return prompt


def recognize(fileNameToRecognize):
    print("Recognizing Audio: " + fileNameToRecognize)
    text = ""
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
  print("handleVoiceCommand ")
  outPath = "audio/hello.mp3"
  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "audio/wav")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes(outPath, "utf-8"))
  return hopeSettings


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
  hope.httpRequest.send_response(200)
  hope.httpRequest.send_header("Content-type", "text/plain")
  hope.httpRequest.end_headers()
  hope.httpRequest.wfile.write(bytes("success","utf-8"))
  return hopeSettings

#Request was for a file, reqturn it
def handleFileRequest(hope, hopeSettings):
  import mimetypes
  import pathlib
  from mimetypes import MimeTypes
  import urllib 


  global loadHTMLTemplate
  

  print("file requested = '" + hope.commandString + "'" )
  # This is a file
  # mime = mimetypes.guess_type(hope.commandString, strict=True)
  mimeType = "audio/mpeg"
  if hope.commandString == "":
      hope.commandString = "index.html"

  try:
    import os

    file_extension = pathlib.Path(hope.commandString).suffix
    print("File Extension: ", file_extension)
    mime = mimetypes.MimeTypes()
    print("mime: ", mime)
    print("hope.commandString: ", hope.commandString)
    # url = urllib.pathname2url(pathlib.Path(hope.commandString))
    # print("URL: ", url)
    print("Mime Type: ", mimeType[0])
      
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(dir_path)
    file_path = os.path.join(dir_path, hope.commandString)
    print(file_path)
    if file_extension == ".glb":
      mimeType = "model/gltf-binary"
    elif file_extension == ".css":
      mimeType = "text/css"
    else:
      mimeType = mime.guess_type(file_path)[0]
    
    print("Sending file + " + file_path + " , '" + mimeType + "'")
    hope.httpRequest.send_response(200)
    hope.httpRequest.send_header("Content-type", mimeType)
    hope.httpRequest.end_headers()
    
    in_file = open(file_path, "rb") # opening for [r]eading as [b]inary
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
    
    # saveHope(hopeSettings)
  except Exception as e:
      pass
      print(str(e))

#############################################################
#  Main
#############################################################

#Resote settings if this is firsat launch
# if gHopeSettings.isInitialized == False:
#   gHopeSettings = restoreHope(gHopeSettings)


#Handle the command, pass in the globals


gHopeSettings = HopeSettings()
gHopeSettings.isInitialized = False
gHopeSettings.hostName = "0.0.0.0"
gHopeSettings.serverPort = 8080
gHopeSettings.commandString = ""
gHopeSettings.audioPath = 'audio/hello.mp3'
gHopeSettings.defaultPrompt = "Type here and click Submit to speak to Hope."
gHopeSettings.defaultResponse = "Output from Hope will appear here."
gHopeSettings.defaultAudioPath = 'audio/hello.mp3'
gHopeSettings.prompt = "Type here and click Submit to speak to Hope."
gHopeSettings.responseText = "Output from Hope will appear here."
gHopeSettings.savePath = "./Hope.json"
gHopeSettings.voiceID = 2330

class HopeSession(object):
    pass
gHope = HopeSession()
# gHope.tokenizer = None #AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
# gHope.model = None #AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
# gHope.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True)
# gHope.textPipeline =  HuggingFacePipeline(pipeline=gHope.generate_text)
# gHope.prompt_with_context = PromptTemplate( input_variables=["instruction", "context"], template="{instruction}\n\nInput:\n{context}")
# gHope.llm_context_chain = LLMChain(llm=gHope.textPipeline, prompt=gHope.prompt_with_context)


gHope.httpRequest = None
# gHope.modelVoice = Model("vosk_models/en")
# gHope.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# gHope.speakModel = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# gHope.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# gHope.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#Windows
# os.chdir("/mnt/g/Source/Hope")

#Mac
# os.chdir("/Users/falco/Documents/Source/Hope/pythonServer")

def doCommand(httpRequest):
    global gHope, gHopeSettings
    gHope.commandString = unquote(httpRequest.path[1:])
    gHope.httpRequest = httpRequest
    HandleHopeCommand(gHope, gHopeSettings)

def handle_post(self, file_path):
    boundary = self.headers.plisttext.split("=")[1]
    remaining_bytes = int(self.headers['content-length'])
    line = self.rfile.readline()
    remaining_bytes -= len(line)
    if boundary not in line:
        return False, "Content NOT begin with boundary"
    line = self.rfile.readline()
    remaining_bytes -= len(line)

    line = self.rfile.readline()
    if not line:
        remaining_bytes -= len(line)

    with open(file_path, 'wb') as out:
        preline = self.rfile.readline()
        remaining_bytes -= len(preline)
        while remaining_bytes > 0:
            line = self.rfile.readline()
            remaining_bytes -= len(line)
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith('\r'):
                    preline = preline[0:-1]
                out.write(preline)
                return True, "File '%s' upload success!" % file_name
            else:
                out.write(preline)
                preline = line
        return False, "Unexpect Ends of data."
        
class MyServer(BaseHTTPRequestHandler):
            
    def do_GET(self):
        doCommand(self)

    def do_PUT(self):
        from pathlib import Path
        print (self.headers)
#        path = self.translate_path(self.path)
#        file_len = len(file_data)
        print(self.headers['filename'])
        file_path = os.path.join(Path.home(), "Documents", "Reality Server", "Uploads", self.headers['filename'])
        print(file_path)
        dir_path = os.path.dirname(file_path)
        length = int(self.headers['Content-Length'])
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as dst:
            dst.write(self.rfile.read(length))

    def do_POST(self):
        print ("MY SERVER: I got a POST request.")
        if self.path == '/verify':
           # Parse the form data posted
          print ("cgi")
          import cgi
          form = cgi.FieldStorage(
              fp=self.rfile, 
              headers=self.headers,
              environ={'REQUEST_METHOD':'POST',
                      'CONTENT_TYPE':self.headers['Content-Type'],
                      })
          print ("form")

          # Begin the response
          self.send_response(200)
          self.end_headers()
          self.wfile.write(bytes('Client: %s\n' % str(self.client_address), "utf-8"))
          self.wfile.write(bytes('User-agent: %s\n' % str(self.headers['user-agent']), "utf-8"))
          self.wfile.write(bytes('Path: %s\n' % self.path, "utf-8"))
          self.wfile.write(bytes('Form data:\n', "utf-8"))

          # Echo back information about what was posted in the form
          for field in form.keys():
              field_item = form[field]
              # print(field_item)
              if field_item.filename:
                  # The field contains an uploaded file
                  from pathlib import Path
                  file_data = field_item.file.read()
                  file_len = len(file_data)
                  file_path = os.path.join(Path.home(), "Documents", "Reality Server", "Uploads", field_item.filename)
                  dir_path = os.path.dirname(file_path)

                  print(dir_path)
                  os.makedirs(dir_path, exist_ok=True)
                  with open(file_path, 'wb') as out:
                      out.write(file_data)

                  del file_data
                  self.wfile.write(bytes('\tUploaded %s as "%s" (%d bytes)\n' % \
                          (field, field_item.filename, file_len), "utf-8"))
              else:
                  # Regular form value
                  self.wfile.write(bytes('\t%s=%s\n' % (field, form[field].value), "utf-8"))
          return
              
                       
    
        
def startServer():
    webServer = HTTPServer((gHopeSettings.hostName, gHopeSettings.serverPort), MyServer)
    print("Server started http://%s:%s" % (gHopeSettings.hostName, gHopeSettings.serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
       
if __name__ == "__main__":
    startServer()
