# To extract the estimated background frame, i have calculated the median frame by taking 25 random frames of the video.
# To get the Detected Moving Pixels in Binary Mask, I subtracted the median frame or the background using cv2.absdiff method.
# I used a threshold in the above image to remove noise and binarize the output and then multiplied the original frame with the binary mask to get 
# the moving pixels in original colour.
# I concatenated the frames both horizontally and vertically to get all the four frames in a single window.
 


import numpy as np
import cv2
import sys

def task1(video):
    # Open Video
    cap = cv2.VideoCapture(video)

    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)

    # Calculate the median along the time axis
    # This is the Estimated Background
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

    # Reset frame number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Convert background to grayscale
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    # Loop over all frames
    ret = True
    while(ret):
      # Read frame
        ret, frame = cap.read()
        if(frame is None):
            break
        #Converting to Grayscale
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Calculate absolute difference of current frame and 
      # the median frame
        dframe = cv2.absdiff(gframe, grayMedianFrame)
      # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
        # Detected Moving Pixels in Binary Mask
        eframe = cv2.cvtColor(dframe, cv2.COLOR_GRAY2BGR)

        orig = frame.copy()
        orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Moving objects in original color with black background
        new_image = orig * (eframe.astype(orig.dtype))

        row1 = np.concatenate((frame, medianFrame), axis=1)
        row2 = np.concatenate((eframe, new_image), axis=1)
        final = np.concatenate((row1, row2), axis=0)


      # Display final window
        cv2.imshow('frame', final)
        key = cv2.waitKey(25)

        #Press 'q' to exit
        if key == ord('q'):
            break

    # Release video object
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    if(sys.argv[1]=="-b"):
        task1(sys.argv[2])