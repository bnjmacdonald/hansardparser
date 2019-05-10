# README

Dockerized HTTP service for parsing Kenya Hansard transcripts. Receives a Hansard PDF or TXT file as input.

`plenaryparser` implements the HTTP service for parsing Hansard plenary transcripts (e.g. [http://parliament.go.ke/the-national-assembly/house-business/hansard](http://parliament.go.ke/the-national-assembly/house-business/hansard)))

`committeeparser` parses Hansard committee transcripts. This is not implemented yet.


## Setup

```sh
pip install git+https://github.com/bnjmacdonald/hansardparser
```


## Examples

### Run Docker network

TODO: ...

### Parse a transcript via POST request

```{python}
import requests
import os
import json
import io

# Text transcript
f = io.StringIO('''<Header>ORAL ANSWERS TO QUESTIONS </Header>\n
Question No.799\n
<Newspeech>MR. SPEAKER: Mr. Ekidoronot in? Next Question.</Newspeech>\n
Question No.780\n
DR. MANTO asked the Minister for Agriculture :-\n
(a)	 whether he is aware that the demand for sugar will be greater than its .production by 1990; and\n
(b) whether he will, therefore, reconsider\n
the suspended plan to establish an additional sugar factory in Busia District.
''')

# POST request
url = "http://localhost:8000"
res = requests.post(url, files={'file': f}, params={filetype': 'txt', 'line_type_labeler': 'supervised', 'line_speaker_span_labeler': 'hybrid'})
assert res.status_code == 200
data = json.loads(res.text)
print(data)
```

## Contributing

If you are interested in contributing to this project, contact @bnjmacdonald (bnjmacdonald@gmail.com).
