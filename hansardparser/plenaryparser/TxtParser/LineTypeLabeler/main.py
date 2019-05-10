"""

Reference for Google ML online predictions: https://cloud.google.com/ml-engine/docs/tensorflow/online-predict
Reference for Tensorflow serving: https://www.tensorflow.org/tfx/serving/api_rest

Http example::

    $ URL=https://us-central1-hansardparser.cloudfunctions.net/predict_hansard_line_type4
    $ MODEL_NAME=hansard_line_type4_lstm_seq2seq_attention_bidirectional_encoder_lstm_attention
    $ curl -d '{"instances": [{"inputs": "MECHANICAL BASE", "context": "ADDITIONAL GRADER FOR RUIRU \nMr. Deputy Speaker: "}]}' \
        -H "Content-Type: application/json" -X POST $URL?model_name=$MODEL_NAME

Example::

    >>> import requests
    >>> import json
    >>> url = "https://us-central1-hansardparser.cloudfunctions.net/predict_hansard_line_type4"
    >>> model_name = "hansard_line_type4_lstm_seq2seq_attention_bidirectional_encoder_lstm_attention"
    >>> instances = [{"inputs": "MECHANICAL BASE", "context": "ADDITIONAL GRADER FOR RUIRU \nMr. Deputy Speaker: "}]
    >>> res = requests.post(url,
    >>>                     params={"model_name": model_name},
    >>>                     headers={"Content-Type": "application/json"},
    >>>                     data={"instances": instances})
    >>> json_data = json.loads(res.content)
    >>> preds = [pred['outputs'][0][0][0] for pred in json_data]
    >>> class_labels = problem.class_labels(None)
"""

import json
import base64
from flask import escape, jsonify, abort
import googleapiclient

from t2t_problem.hansard_line_type4 import HansardLineType4
from tensor2tensor.data_generators.generator_utils import to_example
from tensor2tensor.data_generators.text_problems import VocabType

# google cloud project name.
PROJECT_NAME = "hansardparser"
# default model name, if not provided in the request.
MODEL_NAME = "hansard_line_type4_lstm_seq2seq_attention_bidirectional_encoder_lstm_attention"

def predict_hansard_line_type4(request):
    """HTTP Cloud Function.
    
    Args:

        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    
    Returns:

        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    if request.method == 'POST':
        assert 'content-type' in request.headers and request.headers['content-type'] == 'application/json'
        # retrieve problem encoder.
        problem = HansardLineType4()
        assert problem.vocab_type == VocabType.CHARACTER
        encoder = problem.get_or_create_vocab(data_dir=None, tmp_dir=None)
        # serialize raw string instances
        request_json = request.get_json(silent=True)
        request_args = request.args
        raw_instances = request_json['instances']
        model_name = escape(request_args['model_name']) if 'model_name' in request_args else MODEL_NAME
        serialized_instances = []
        for instance in raw_instances:
            if isinstance(instance, str):
                instance = {'inputs': instance}
            inputs = escape(instance['inputs'])
            context = escape(instance['context']) if 'context' in instance else ''
            encoded_instance = problem.encode_example({"inputs": inputs, "context": context, "label": 0}, encoder)
            # encoded_sample.pop('targets')
            # encoded_sample.pop('context')
            serialized_instance = to_example(encoded_instance).SerializeToString()
            serialized_instances.append(serialized_instance)
        instances = []
        for serialized_instance in serialized_instances:
            instances.append({"b64": base64.b64encode(serialized_instance).decode('utf-8')})
        # model_name = "hansard_line_type4_predict_lstm_encoder_lstm_attention"
        # url = "https://us-central1-hansardparser.cloudfunctions.net/predictHansardLineType4"
        # headers = {"Content-Type": "application/json"}
        # makes predict request.
        predictions = _predict_json(PROJECT_NAME, model_name, instances, version=None)
        return jsonify(predictions)
    return abort(405)



def _predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:

        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.

    Returns:

        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'
    if version is not None:
        name += f'/versions/{version}'
    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']
