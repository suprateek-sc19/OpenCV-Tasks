# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# split method for Task 1
def split(color_space, Image):
    # reading the image input by the user
    image = cv2.imread(Image)
    # converting image to rgb color space as OpenCV default color space is BGR
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # converting image to color space specified by the user using if-else statements
    if color_space == '-XYZ':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    elif color_space == '-Lab':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif color_space == '-YCrCb':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == '-HSB':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        print("Wrong Argument")
        return
    
    # After the image is coverted to specified color space, we split it into its 3 color components using cv2.split
    x,y,z = cv2.split(img)
    
    # Making a plot of 4 subplots to generate the required output
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(19.20,10.80))
    axes[0, 0].imshow(rgb)
    axes[0, 0].title.set_text('Original Image')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(x, cmap="gray")
    axes[0, 1].title.set_text('C1')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(y, cmap="gray")
    axes[1, 0].title.set_text('C2')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(z, cmap="gray")
    axes[1, 1].title.set_text('C3')
    axes[1,1].axis('off')
    # Saving the plot as an image in 1920x1080 size
    plt.savefig('colorspaces.png', dpi = 100)
    # Reading the saved image using cv2
    final = cv2.imread('colorspaces.png')
    
    # Display the final result in a new window
    cv2.imshow('COLOR SPACES',final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# method to combine the two images (scenic and greenScreen image)
def combine(greenScreenImage, scenicImage):
    # Read in the images
    image = cv2.imread(greenScreenImage)
    background = cv2.imread(scenicImage)
    # get the height and width of scenic image as the output image should of the same size as the scenic image
    height = background.shape[0]
    width = background.shape[1]

    # Making a copy of the image
    image_copy = np.copy(image)
    # Upcaling the image to size of scenic image
    image_copy = cv2.resize(image, (width, height),interpolation = cv2.INTER_LINEAR)
    # Changing color to RGB (from BGR)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

     # creating a colour threshold to remove the desired green region
    lower_green = np.array([0, 100, 0])    
    upper_green = np.array([126, 255, 100])

    # Defining the masked area
    mask = cv2.inRange(image_copy, lower_green, upper_green)

    # Masking the image to let the person show through
    masked_image = np.copy(image_copy)
    masked_image[mask != 0] = [0, 0, 0]

    #converting background it to RGB 
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background_image = np.copy(background)
    
    # Masking the background so that the pizza area is blocked
    background_image[mask == 0] = [0, 0, 0]

    # Adding the two images together to create a complete image
    final_image = background_image + masked_image
    
    # creating a white background image
    white = np.zeros([height,width,3],dtype=np.uint8)
    white.fill(255)
    white[mask == 0] = [0, 0, 0]

    # creating image of person with white background
    white_background = white + masked_image
    
    # plotting the 4 subplots to generate the required output
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(19.20,10.80))
    axes[0, 0].imshow(image_copy)
    axes[0, 0].title.set_text('Photo of a person with green screen')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(white_background)
    axes[0, 1].title.set_text('Photo of the same person with white background')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(background)
    axes[1, 0].title.set_text('Scenic photo')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(final_image)
    axes[1, 1].title.set_text('The same person in the scenic photo')
    axes[1,1].axis('off')
    # saving the plot to an 1920x1080 image
    plt.savefig('chroma.png', dpi = 100)

    # reading the saved image
    final = cv2.imread('chroma.png')
    
    # Display the final result
    cv2.imshow('CHROMA KEY',final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Parsing command line arguments and calling the appropriate methods 
if __name__ == "__main__": 
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    if(arg1 == "-XYZ" or arg1 == "-Lab" or arg1 == "-YCrCb" or arg1 == "-HSB"):
        split(arg1, arg2)
    else:
        combine(arg1, arg2) 