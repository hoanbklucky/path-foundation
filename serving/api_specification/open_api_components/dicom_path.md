# dicom_path

## Properties

Name               | Type       | Description                                                                                                                                                                               | Notes
------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----
**series\_path**   | **String** | The path to a DICOM Series in a DICOMWeb Store. Provide the URI in this format: https://{DICOMWEB-STORE-URI}/studies/{STUDY-UID}/series/{SERIES-UID}                                      | [default to null]
**instance\_uids** | **List**   | A list of unique identifiers for DICOM SOP Instances that contain the image pixels corresponding to the specified coordinates. All SOP Instances listed must have the same pixel spacing. | [default to null]

[[Back to payload schema]](../README.md#payload-schema)
