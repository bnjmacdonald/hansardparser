"""converts audio files to video files.

Uses ffmpeg for conversion.

Used for converting Hansard committee transcripts from audio to video
so that they can be uploaded to Youtube (privately) and transcribed.

"""

from hansardlytics import settings

import os
import subprocess


inpath = '/Users/bnjmacdonald/downloads'
outpath = '/Users/bnjmacdonald/downloads'

img_fname = os.path.join(settings.MEDIA_ROOT, 'temp', 'placeholder.png')
audio_fname = 'DC LABOUR & SOCIAL WELFARE  21-03-2017(MANAGEMENT OF BIDCO)A.MP3'
out_fname = 'out_short.avi'

cmd = 'ffmpeg -y -loop 1 -r 1 -i "{0}" -ss 00:01:00 -i "{1}" -to 00:02:00 -c:a copy -shortest "{2}"'.format(img_fname, os.path.join(inpath, audio_fname), os.path.join(outpath, out_fname))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

# out = p.stdout.read()
error = p.stderr.read()
if error:
    raise Exception(error)
out = out.decode('utf-8')  # windows-1252
