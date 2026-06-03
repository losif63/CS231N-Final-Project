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