from http.server import BaseHTTPRequestHandler, HTTPServer
import torch, os
from urllib.parse import unquote
from vosk import Model

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
from transformers import SpeechT5HifiGan
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline



print("Cuda = " + str(torch.cuda.is_available()))

class HopeSettings(object):
    pass

gHopeSettings = HopeSettings()
gHopeSettings.isInitialized = False
gHopeSettings.hostName = "0.0.0.0"
gHopeSettings.serverPort = 8080
gHopeSettings.commandString = ""
gHopeSettings.audioPath = './audio/pcvoice.mp3'
gHopeSettings.defaultPrompt = "Type here and click Submit to speak to Hope."
gHopeSettings.defaultResponse = "Output from Hope will appear here."
gHopeSettings.defaultAudioPath = './audio/pcvoice.mp3'
gHopeSettings.prompt = "Type here and click Submit to speak to Hope."
gHopeSettings.responseText = "Output from Hope will appear here."
gHopeSettings.savePath = "./Hope.json"
gHopeSettings.voiceID = 2330

class HopeSession(object):
    pass
gHope = HopeSession()
gHope.tokenizer = None #AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
gHope.model = None #AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
gHope.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True)
gHope.textPipeline =  HuggingFacePipeline(pipeline=gHope.generate_text)
gHope.prompt_with_context = PromptTemplate( input_variables=["instruction", "context"], template="{instruction}\n\nInput:\n{context}")
gHope.llm_context_chain = LLMChain(llm=gHope.textPipeline, prompt=gHope.prompt_with_context)


gHope.httpRequest = None
gHope.modelVoice = Model("vosk_models/en")
gHope.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
gHope.speakModel = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
gHope.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
gHope.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

#Windows
# os.chdir("/mnt/g/Source/Hope")

#Mac
os.chdir("/Users/falco/Documents/Source/Hope/pythonServer")

def doCommand(httpRequest):
    global gHope
    gHope.commandString = unquote(httpRequest.path[1:])
    gHope.httpRequest = httpRequest
    code = ""
    with open("./python/Hope.py", mode="r", encoding="utf-8") as hello:
      code = hello.read()
    # print(code)
    exec(code)

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):     
        doCommand(self)
       
if __name__ == "__main__":        
  
    webServer = HTTPServer((gHopeSettings.hostName, gHopeSettings.serverPort), MyServer)
    print("Server started http://%s:%s" % (gHopeSettings.hostName, gHopeSettings.serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
