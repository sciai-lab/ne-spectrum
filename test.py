from ne_spectrum import TSNESpectrum, CNESpectrum
import os
import numpy as np

# get random data
x1 = np.random.randn(1000, 10)
x2 = np.random.randn(1000, 10) + np.ones((1000, 10))
x = np.concatenate([x1, x2])


for spectrum in [TSNESpectrum(num_slides=10), CNESpectrum(num_slides=10)]:
    spectrum.fit(x)

    video_file = "test.mp4"
    spectrum.save_video(save_path="./", file_name=video_file)
    os.remove(video_file)
