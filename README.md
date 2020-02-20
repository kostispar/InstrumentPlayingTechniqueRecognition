# InstrumentPlayingTechniqueRecognition
Instrument Playing Technique Recognition


Command line use of the classifier.py

```python classifier.py  <path_to_pretrained_model> <path_to_audio_recording> <path_to_ground_truth_file>```

The first argument is the path to the pretrained model (available at models directory of this repo). The seconds argument is the path to the audio file recording. The third argument refers to a .segment file and is optional. 

Classifier.py will extract the audio features of the given recording for mt_window , mt_step , st_window and st_step equal to the respective values of the trained model passed. Then, the pretrained classifier will be loaded. Predictions will be made on the segments of the recording. The output is a single plot showing the baseline classification results.
