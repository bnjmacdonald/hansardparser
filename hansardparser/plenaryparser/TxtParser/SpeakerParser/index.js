/**
 * HTTP Cloud Function: predictHansardLineType4
 * 
 * This function returns a prediction for each line in a Hansard transcript regarding
 * whether or not the line contains a speaker name.
 * Accepts a POST request with an array of transcript lines and returns the
 * prediction for each line.
 * 
 * @param {Object} req Cloud Function request context.
 * @param {Object} res Cloud Function response context.
 * 
 * Based on:
 *    https://github.com/jdabello/hello-ml-engine-predict/blob/master/cloudfunctions/index.js
 * 
 * Todos:
 *  
 *    TODO: return label_codes with response.
 *
 * EXAMPLES::
 * 
 *    curl -d '{"instances": ["Question no. 238", "Payment of Retirement Benefits to Mr. Chemwengut", "committee because we must leave room for people to "]}' \
 *      -H "Content-Type: application/json" -X POST https://us-central1-hansardparser.cloudfunctions.net/predictHansardLineType4
 */

var google = require('googleapis').google;

const MODEL_NAME = "hansard_line_has_speaker_predict_transformer_encoder_transformer_small";
const PROJECT_NAME = "hansardparser";

exports.predictHansardLineHasSpeaker = (req, res) => {
  var ml = google.ml('v1');
  google.auth.getApplicationDefault(function (err, authClient, projectId) {
    if (err) {
      console.log('Authentication failed because of ', err);
      return;
    }
    if (authClient.createScopedRequired && authClient.createScopedRequired()) {
      var scopes = ['https://www.googleapis.com/auth/cloud-platform'];
      authClient = authClient.createScoped(scopes);
    }
    if (req.method == 'POST' && req.get('content-type') == 'application/json') {
      var resourceData = req.body;
      console.log(resourceData);
      ml.projects.predict({
        name: `projects/${PROJECT_NAME}/models/${MODEL_NAME}`,
        auth: authClient,
        resource: resourceData
      }, function(err, result) {
        if (err) {
          console.error(`Error in making predictions: ${err}`);
        } else {
          console.log(result);
          res.status(200).send(JSON.stringify(result.data.predictions) || '').end();
        }
      })
    } else {
      res.status(405).send({error: 'This endpoint only accepts PUT requests with application/json body.'});
    }
  })
};
