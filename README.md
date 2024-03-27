Libraries to be installed-
1)pydub
2)librosa
3)numpy
4)matplotlib
5)tensorflow
6)os

Just use pip install <package_name> to install these libraries

Before training the models, make sure to run the script "convert_train_val_audio_to_image.py" . This ensures that the audio files are converted to spectrograms which is what the model requires as input.
Please make sure to initialize the directory path "AUDIO_DATASET_PATH" while converting the audio to image.

If you want to run the training scripts, then please initialize the train and test directories for the spectrograms as they are generated in the "convert_train_val_audio_to_image.py" file.

In the testing script, "TEST_DATA_DIRECTORY_ABSOLUTE_PATH" and "OUTPUT_CSV_ABSOLUTE_PATH" must be set before running the script. In case you want to use the test files or output.csv already provided, set the path of that.

Audio dataset is not provided, but you can use any relevant dataset. The models have been trained on a dataset which has 13 classes. Only the .keras file for mobileNet architecture is provided as the weights for ResNet are too large. Please make relevant changes to the model bases on your dataset.

Ideal order of running scripts-
1)convert_train_val_audio_to_image.py
2)training_mobileNet.py / training_resNet.py
3)testing_code_mobileNet.py / testing_code_resNet.py


