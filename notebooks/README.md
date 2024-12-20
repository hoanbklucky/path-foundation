# Path Foundation Notebooks

*   [Quick start with Hugging Face](quick_start_with_hugging_face.ipynb) -
    Example of encoding a pathology image patch into an embedding vector by
    running the model locally from Hugging Face.

*   [Quick start with Vertex Model Garden](quick_start_with_model_garden.ipynb) -
    Example of serving the model on
    [Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/overview)
    and using Vertex AI APIs to encode pathology image patches to embeddings in
    online or batch workflows.

*   [Train a data efficient classifier - GCS version](train_data_efficient_classifier_gcs.ipynb) -
    Example of using the generated embeddings to train a custom classifier with
    less data and compute. This version shows how to use the data as files in
    [Cloud Storage (GCS)](https://cloud.google.com/storage).

*   [Train a data efficient classifier - DICOMWeb version](train_data_efficient_classifier_dicom.ipynb) -
    Example of using the generated embeddings to train a custom classifier with
    less data and compute. This version shows how to use the data as DICOM
    objects in
    [Cloud DICOM store](https://cloud.google.com/healthcare-api/docs/how-tos/dicom).

*   [Simplify client code with EZ-WSI](simplify_client_code_with_ez_wsi.ipynb) -
    Instructions how to utilize
    [EZ-WSI DicomWeb library](https://github.com/GoogleCloudPlatform/EZ-WSI-DICOMweb)
    to simplify client side code for working with DICOM data and generating
    embeddings from a variety of data sources including Cloud DICOM store, GCS,
    and locally stored files or in-memory data representations.

*   [Fine-tune data efficient classifier](fine_tune_data_efficient_classifier.ipynb)
    Example of fine-tuning the weights of the pathology embedding model to
    classify pathology image patches as an alternative to a linear classifier.