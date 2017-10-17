"""uploads video to youtube and (optionally) downloads caption track.

TODO:
* add video upload capability (see https://developers.google.com/youtube/v3/docs/videos/insert)
* automatically download caption track once it has been created. (see https://developers.google.com/youtube/v3/docs/captions)
"""

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser
from config import DEVELOPER_KEY

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# TODO...