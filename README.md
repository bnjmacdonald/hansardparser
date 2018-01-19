# README

Contains methods for parsing Kenya Hansard transcripts. Takes a Hansard PDF file as input (e.g. [http://parliament.go.ke/the-national-assembly/house-business/hansard](http://parliament.go.ke/the-national-assembly/house-business/hansard)) and returns a list of json documents, where each document represents a `speech` (including fields for name of the speaker, speech text, heading, ...). 

The current release relies entirely on regular expressions to parse PDF files. We are currently re-implementing the library using deep learning so that it is possible to accurately parse Hansards for many countries with ease.

## Contributing

If you are interested in contributing to this project, contact @bnjmacdonald (bnjmacdonald@gmail.com). Since this project is closely related to a larger project on citizen engagement in politics, the Github issues describing next steps are located in a different github repository.
