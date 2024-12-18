# Path Foundation serving

This folder contains the source code and configuration necessary to serve the
model on
[Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container).
The implementation follows this
[container architecture](https://developers.google.com/health-ai-developer-foundations/model-serving/container-architecture).

The serving container can be used in both online and batch prediction workflows:

*   **Online predictions**: Deploy the container as a
    [REST](https://en.wikipedia.org/wiki/REST) endpoint, like a
    [Vertex AI endpoint](https://cloud.google.com/vertex-ai/docs/general/deployment).
    This allows you to access the model for real-time predictions via the REST
    [Application Programming Interface (API)](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/serving-api).

*   **Batch predictions**: Use the container to run large-scale
    [Vertex AI batch prediction jobs](https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions)
    to process many inputs at once.

Note: PETE is an acronym used throughout the code that stands for Pathology
Encoder Tech Engine.

## Description of select files and folders

*   [`serving_framework/`](./serving_framework): A library for
    implementing Vertex AI-compatible HTTP servers.

*   [`vertex_schemata/`](./vertex_schemata): Folder containing YAML files that
    define the
    [PredictSchemata](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/PredictSchemata)
    for Vertex AI endpoints.

*   [`abstract_pete_predictor.py`](./abstract_pete_predictor.py): Defines
    `AbstractPetePredictor`, an abstract base class that provides a blueprint
    for PETE predictor classes. Subclasses must implement the predict method to
    provide concrete prediction logic.

*   [`Dockerfile`](./Dockerfile): Defines the Docker image for serving the
    model.

*   [`entrypoint.sh`](./entrypoint.sh): A bash script that is used as the Docker
    entrypoint. It sets up the necessary environment variables, copies the
    TensorFlow [SavedModel(s)](https://www.tensorflow.org/guide/saved_model)
    locally and launches the TensorFlow server and the frontend HTTP server.

*   [`pete_error_mapping.py`](./pete_error_mapping.py): Defines mappings between
    errors in Python and error codes returned in API responses.

*   [`pete_errors.py`](./pete_errors.py): Defines error classes. It includes the
    base class `PeteError` and various specific error classes that inherit from
    it.

*   [`pete_flags.py`](./pete_flags.py): Defines flags configured by
    environmental variables that configure container.

*   [`pete_icc_profile_cache.py`](./pete_icc_profile_cache.py): Enables
    [ICC profile](https://en.wikipedia.org/wiki/ICC_profile) caching using
    [Redis](https://redis.io) or
    [Cloud Storage](https://cloud.google.com/storage).

*   [`pete_prediction_executor.py`](./pete_prediction_executor.py): Defines the
    prediction executor for PETE. It includes the main function that runs the
    prediction executor loop.

*   [`pete_predictor_v2.py`](./pete_predictor_v2.py): Defines the PETE predictor
    v2 class. It includes the predict function that runs inference on provided
    patches.

*   [`requirements.txt`](./requirements.txt): Lists the required Python
    packages.

*   [`server_gunicorn.py`](./server_gunicorn.py): Creates the HTTP server that
    launches the prediction executor.

## Dependencies

*   [`data_processing/`](../data_processing): A library for data
    retrieval and processing.
