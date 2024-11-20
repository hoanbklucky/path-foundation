# Path Foundation Serving

This folder contains the container implementation for serving the Path
Foundation model. It includes the Dockerfile, requirements files, and Python
scripts necessary to run the model server and REST API.

For documentation on its API, see the
[API Specification](../docs/api_specification/README.md)

## Description for select files and folders

*   [`Dockerfile`](./Dockerfile): This file defines the Docker image that will
    be used to serve the model. It includes the necessary dependencies, such as
    TensorFlow and the model server.
*   [`requirements.txt`](./requirements.txt): This file lists the Python
    packages that are required to run the model server.
*   [`abstract_pete_predictor.py`](./abstract_pete_predictor.py): This file
    defines the abstract base class `AbstractPetePredictor` for embeddings
    predictors. It includes the abstract method `predict` that needs to be
    implemented by concrete PETE predictor classes.
*   [`divided_image_launcher.sh`](./divided_image_launcher.sh): This file is a
    bash script that launches the divided image for the embeddings model. It
    sets up the necessary environment variables and launches the inference
    engine, the REST API server, and the front end.
*   [`entrypoint.sh`](./entrypoint.sh): This file is a bash script that serves
    as the entrypoint for the Docker container. It launches the model server and
    the front end.
*   [`pete_error_mapping.py`](./pete_error_mapping.py): This file defines
    mappings between errors in Python and error codes returned in API responses.
    It includes a dictionary that maps `PeteError` types to `ErrorCode` values.
*   [`pete_errors.py`](./pete_errors.py): This file defines error classes. It
    includes the base class `PeteError` and various specific error classes that
    inherit from it.
*   [`pete_flags.py`](./pete_flags.py): This file defines flags configured by
    environmental variables that configure the pathology embeddings prediction
    container.
*   [`pete_icc_profile_cache.py`](./pete_icc_profile_cache.py): This file
    defines the ICC profile cache. It includes functions for caching ICC
    profiles in Redis and GCS.
*   [`pete_logging.py`](./pete_logging.py): This file defines the logging
    functionality for pathology embeddings. It includes functions for
    initializing logging and setting the log signature.
*   [`pete_prediction_executor.py`](./pete_prediction_executor.py): This file
    defines the prediction executor for PETE. It includes the main function that
    runs the prediction executor loop.
*   [`pete_predictor_v2.py`](./pete_predictor_v2.py): This file defines the PETE
    predictor v2 class. It includes the predict function that runs inference on
    provided patches.
*   [`requirements.txt`](./requirements.txt): This file lists the Python
    packages that are required to run the model server.
*   [`server_gunicorn.py`](./server_gunicorn.py): This file defines the gunicorn
    server for PETE. It includes the main function that launches the gunicorn
    application.
*   [`serving_requirements.txt`](./serving_requirements.txt): This file lists
    the Python packages that are required to run the model server.
*   [`api_specification`](./api_specification): The files define the API
    specification for a pathology embedding service, including the request and
    response schema, error codes, and validation rules.
*   [`data_models`](./data_models): The files define data models and associated
    tests for embedding requests and responses, including patch coordinate
    handling and conversion to/from JSON format.
