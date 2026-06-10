# Project AI Usage

For this project, our team used Claude Code (Sonnet 4.6, Opus 4.8, and Haiku 4.5) to assist in writing the code for many parts of our project.

Files where Claude Code was used contain "Written using Claude Code" in their header.

No parts of the report or poster were written by an LLM.

Below is a variety of prompts given to Claude Code for this project.

# Prompts

```
Please follow the instructions in /cnn_baseline to create a baseline CNN model for this project. To reiterate, here are the instructions to complete the task for the /cnn_baseline folder:

"""
This folder should contain simple CNN model used as a baseline.

There will be a /levels folder and a /videos folder. Please look at /gemometry_dash for how the "levels" folder is formatted and /data_collection for how the /videos folder is formatted. The /videos folder will contain the videos (inputs) and the /levels folder will have a difficulty rating (int 1-10) for each level and corresponding video, which is the label we're trying to predict. Jusst for context, the videos are about 1-3 minutes long. (Note the videos aren't always the same length.)

You should create a simple model to predict the difficulty of the videos.

This model should work as follows:

- Run the CNN (resnet backbone) over a frame of the video every 4 seconds (1/4 FPS)
- Mean all the CNN features together
- Flatten the CNN features
- Have a fully connected network convert the CNN features into a difficulty prediction (cross-entropy loss)

You should make everything adjustable, including the FCN architecture, the resnet version, and the cropping of the video (should crop a square from the center, but might not need to go all the way to the edges. For your initial version, go all the way to the edges.)

You should use PyTorch. Use a Dataset and Dataloader to get the videos. Use open CV for getting the video frames. Update requirements.txt with any requirements you use.
"""
```

```
Please update the code to use a pre-extraction script. Please use 256*256 for the resolution, then leave the ability to downsize later. Each extracted "frame" should be a square cut from the center of the video (leaving off the sides).

Update the dataloader code to also load the data with this change in mind. Make sure the entire cnn_baseline_network folder works after your change by thinking it through and maybe running a VERY small test at the end
```

```
Can you switch the cnn baseline to use Ordinal Classification (Binary Decomposition)? Make all the right and standard changes to make this perform well. The backbone should still be frozen, and keep using the attention-based mean pooling.
```

```
Can you update scripts/check_videos.py to, if broken_videos.txt already exists, to first ask the user if they want to use the existing broken_videos.txt? If they answer no, it should do the scan, if they answer yes, it should skip the scan and going straight to asking the user if they want to delete the videos
```

```
Can you complete the file "scripts/preencode_and_label_videos.py"? This file should do the same thing as preencode_videos.py, but also produce a JSON file containing the model's outputs for each frame (relative to the output frames, so the output is 30 FPS so the given model in an arg should make a height and occlusion prediction for each of those 30 frames per second, but it needs to predict off of the full resolution image (at least resized to 620x620).
Show
```

```
Can you update the scripts/baseline_train.py script to output the val accuracy, val MAE, val off1 (the percent of labels that are within one of the correct label), and those three for the test set also? Specifically right now it's missing the off1 statistic for both
```

```
Here is your task:

We need to create a baseline model. The baseline model is pretty simple: just a ResNet-18 backbone that extracts features from a frame of video every 4 seconds, then aggregates the frames by doing a weighted sum with the weight determined by a Linear layer. After getting the weighted sum of all the ResNet-18 features, flatten the features and use an MLP for the final prediction. The final prediction should be 10 classes for difficulty star rating (1-10). The MLP should use 512 and 256 hidden layers. Also use dropout.

A similar implementation is in cnn_baseline_network, but that code is old and doesn't freeze the backbone.

videos_processed has sub folders called "{N}stars" (e.g. 1stars/, 2stars/, etc) where N is the star rating and that sub folder contains videos with that star rating.

You should add two python files to scripts/

The first should go through the videos in the sub folders of videos_processed and run ResNet-18 over the videos. The results should go in a folder called processed_resnet18/ with the same subfolders and be saved as numpy files. Name this script baseline_extract.py. For this file, do batches of videos from each star difficulty iteratively, such that if the program is stopped half way through, there would be roughly an equal number of numpy files for each star rating. Consider how best to do this. Note that decoding is probably going to be the bottle neck, so consider if you should have multiple workers doing decoding or batching or something. Consider how to optimize this and pick the best approach.

Then, create a script named baseline_train.py that defines and trains the model defined above on the extracted features in processed_resnet18/.

You should use the cv_final_proj conda environment. Also use decord to get the video frames, it should be faster. This code will be run on an M1 Macbook, so use MPS. Please make a plan to do this. Make specific notes of the important details in your plan
```

```
The program is taking over 16GB of RAM. Let's use fewer workers and a smaller prefetch queue and GPU memory pruning
```