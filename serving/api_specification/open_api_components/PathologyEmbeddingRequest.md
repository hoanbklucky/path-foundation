# PathologyEmbeddingRequest

## Properties

Name                   | Type                            | Description                                                                                                                                                                              | Notes
---------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----
**dicom\_path**        | [**dicom_path**](dicom_path.md) |                                                                                                                                                                                          | [optional] [default to null]
**image\_file\_uri**   | **String**                      | The path to an image file in a Google Cloud Storage bucket. Provide the URI in this format: gs://{BUCKET-NAME}/{OPTIONAL-FOLDER-HIERARCHY}/{FILE-NAME}.{FILE-TYPE}                       | [optional] [default to null]
**input\_bytes**       | **byte[]**                      | Input data as a base64-encoded string. Refer to the API specification for details.                                                                                                       | [optional] [default to null]
**bearer\_token**      | **String**                      | The token to access the Cloud DICOM Store or Cloud Storage bucket where the images are stored.                                                                                           | [optional] [default to null]
**patch\_coordinates** | [**List**](patch_coordinate.md) | An array of patch coordinates.                                                                                                                                                           | [default to null]
**extensions**         | **Object**                      | An optional dictionary to enable flexible communication between the client and server. Refer to [extensions](../README.md#extensions) for the list of supported keys and their purposes. | [optional] [default to null]

[[Back to payload schema]](../README.md#payload-schema)
