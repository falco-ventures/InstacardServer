class HopeSession(object):
    pass
gHope = HopeSession()

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset
import torch
from transformers import SpeechT5HifiGan

gHope.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
gHope.speakModel = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
gHope.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
gHope.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def speakTextToFile(voiceID, text, path):
    global gHope

    inputs = gHope.processor(text=text, return_tensors="pt")
    speaker_embeddings = torch.tensor(gHope.embeddings_dataset[voiceID]["xvector"]).unsqueeze(0)

    spectrogram = gHope.speakModel.generate_speech(inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        speech = gHope.vocoder(spectrogram)
        
    import soundfile as sf
    sf.write(fileName, speech.numpy(), samplerate=16000)

maxDatasetID = 739
for datasetID in range(maxDatasetID):
    datasetID = datasetID * 10
    input_text = "Don't count the days, make the days count."
    fileName = "voices/" + str(datasetID) + ".wav"
    speakTextToFile(datasetID, input_text, fileName)
    

# from playsound import playsound
# playsound(fileName)