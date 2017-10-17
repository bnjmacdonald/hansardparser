"""

Two distinct tasks: speech to text and speaker recognition.

Left off:
    - ran samples with IBM Watson, Baidu (online), and Google Speech API. None of them perform very well at all. 

Next steps:
    - try customization of IBM Watson. 

IBM Watson Speech to Text API
-----------------------------

Customization: https://www.ibm.com/watson/developercloud/doc/speech-to-text/custom.html

Tutorial: https://www.ibm.com/watson/developercloud/doc/speech-to-text/tutorial.html

Python example: https://github.com/watson-developer-cloud/python-sdk/blob/master/examples/speech_to_text_v1.py

Pros:
    - customizable.
    - has speaker recognition.
    - many free hours per month.

Google Cloud Speech API
-----------------------

Python code samples: https://github.com/GoogleCloudPlatform/python-docs-samples

Cons:
    - 

Baidu Deep Speech API
---------------------

https://swiftscribe.ai

Also see: https://github.com/baidu-research/warp-ctc and https://github.com/mozilla/DeepSpeech

Pros:
    - easy to edit

Cons:
    - no API?
    - no speaker recognition?

Microsoft
---------

https://msdn.microsoft.com/en-us/library/jj127860.aspx

Dragon
------

https://www.nuance.com/dragon/dragon-for-mac.html

Other
-----

Kaldi: https://github.com/kaldi-asr/kaldi

CMUSphinx: https://cmusphinx.github.io/wiki/download/


Audio file converter
--------------------
http://online-audio-converter.com/

"""

# TESTING
from pydub import AudioSegment
t1 = 10000 #Works in milliseconds
t2 = t1 + 50000
newAudio = AudioSegment.from_wav("/Users/bnjmacdonald/downloads/test1.wav")
newAudio = AudioSegment.from_mp3("/Users/bnjmacdonald/downloads/test1.mp3")
# newAudio = AudioSegment.from_ogg("/Users/bnjmacdonald/downloads/test1.flac")
newAudio = newAudio[t1:t2]
newAudio.export('/Users/bnjmacdonald/downloads/newSong.wav', format="wav") #Exports to a wav file in the current path.


import audiotools

audiotools.open('/Users/bnjmacdonald/downloads/test1.mp3').convert('/Users/bnjmacdonald/downloads/newSong.flac', audiotools.flac.FlacAudio)
f = audiotools.open('/Users/bnjmacdonald/downloads/newSong.flac')
audiotools.PCMConverter(f, channels=1, sample_rate=f.sample_rate(), channel_mask=1, bits_per_sample=f.bits_per_sample())
reader = audiotools.PCMFileReader('/Users/bnjmacdonald/downloads/test1.flac', sample_rate=44100, bits_per_sample=16, channels=1, channel_mask=1)
audiotools.flac.FlacAudio.from_pcm('/Users/bnjmacdonald/downloads/newSong.flac', reader)
audio = reader.read(f)
md = f.get_metadata()
md
f.channels() = 1
f.update_metadata()

import json
import os
import io
from watson_developer_cloud import SpeechToTextV1

username = "" # TODO
password = ""
speech_to_text = SpeechToTextV1(
    username=username,
    password=password,
    x_watson_learning_opt_out=False
)

print(json.dumps(speech_to_text.models(), indent=2))

print(json.dumps(speech_to_text.get_model('en-US_BroadbandModel'), indent=2))

# os.path.join(os.path.dirname(__file__), 
with open('/Users/bnjmacdonald/downloads/newSong.wav',
          'rb') as audio_file:
    # content = audio_file.read()
    # content = content[:int(len(content)/200)]
    # print(len(content))
    # content = io.BufferedReader(io.BytesIO(content))
    result = speech_to_text.recognize(
        audio_file, content_type='audio/wav', timestamps=True,
        word_confidence=True, interim_results=True, speaker_labels=True)  # , 

print(json.dumps(result, indent=2))

# GOOGLE CLOUD API
import io
import os
import json

# Imports the Google Cloud client library
from google.cloud import speech

# from oauth2client.client import GoogleCredentials
# credentials = GoogleCredentials.get_application_default()
# GoogleCredentials.new_from_json(f)

# f = open(GOOGLE_APPLICATION_CREDENTIALS, 'r')
# with open(GOOGLE_APPLICATION_CREDENTIALS, 'r') as f:
#     c = json.load(f)
GOOGLE_APPLICATION_CREDENTIALS='/Users/bnjmacdonald/.ssh/hansardlytics-38f91de7388a.json'

# Instantiates a client
speech_client = speech.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
# speech_client = speech.Client()

# The name of the audio file to transcribe
# file_name = os.path.join(
#     os.path.dirname(__file__),
#     'resources',
#     'audio.raw')
file_name = '/Users/bnjmacdonald/downloads/newSong.flac'

# Loads the audio into memory
with io.open(file_name, 'rb') as audio_file:
    content = audio_file.read()
    # content = content[:int(len(content)/10)]
    # print(len(content))
    sample = speech_client.sample(
        content=content, 
        # stream=audio_file,
        # source_uri=None,
        encoding='FLAC',
        sample_rate_hertz=16000
    )

# Detects speech in the audio file
operation = sample.long_running_recognize('en-US')
alternatives = operation.results
alternatives = sample.recognize('en-US')

for alternative in alternatives:
    print('Transcript: {}'.format(alternative.transcript))
