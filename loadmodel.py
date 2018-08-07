import sys
import numpy as np

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

def loadmodel(model_path):

    # Import custom model in Mask RCNN for detecting notes in spectrograms
    from Mask_RCNN.samples.newtest.notes_config import NotesConfig, get_ax
    
    # print model details
    #NotesConfig().display()

    # modify custom config to use only 1 image and 1 gpu for execution
    class InferenceConfig(NotesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    # Create model in inference (not training) mode
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
                            model_dir="logs")

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    return model