# Exploring Image Processing Techniques with OpenCV and Python 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 0.85 and 1.28 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** SETHUKKARASI C
- **Register Number:** 212223230201

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
img = cv2.imread('Eagle_in_Flight.jpg',0)
```

#### 2. Print the image width, height & Channel.
```
img.shape
```

#### 3. Display the image using matplotlib imshow().
```
plt.imshow(img, cmap ='gray')
plt.axis('off')
plt.title('Eagle In Flight - Grayscale')
plt.show()
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```
cv2.imwrite('Eagle_in_Flight_saved.png', img)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
img_bgr = cv2.imread('Eagle_in_Flight.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```
plt.imshow(img_rgb)
plt.title('Eagle In Flight - RGB')
plt.axis('on')
plt.show()
```

```
img_rgb.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```
crop = img_rgb[25:410,200:545]
#crop = img_rgb[350:390,475:545]
plt.imshow(crop)
plt.title('Eagle In Flight Cropped - RGB')
plt.axis('off')
plt.show()
```

```
crop.shape
```

#### 8. Resize the image up by a factor of 2x.
```
resize = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
resize.shape
```

#### 9. Flip the cropped/resized image horizontally.
```
img_hori = cv2.flip(crop,1)
img_vert = cv2.flip(crop,0)
img_both = cv2.flip(crop,-1)
```

```
plt.figure(figsize=(18,6))
plt.subplot(141); plt.imshow(img_hori)
plt.title("Horizontal Flip")
plt.subplot(142); plt.imshow(img_vert)
plt.title("Vertical Flip")
plt.subplot(143); plt.imshow(img_both)
plt.title("Both Flipped")
plt.subplot(144); plt.imshow(crop)
plt.title("Original")
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```
apollo = cv2.imread('Apollo-11-launch.jpg')

plt.imshow(apollo)
plt.title('Apollo 11 - RGB')
plt.axis('on')
plt.show()
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
text_img = apollo.copy()
f_scale = 2
f_color = (255,255,255)
f_thickness = 3
text_img = cv2.putText(text_img, text, (300,700), font_face, f_scale, f_color, f_thickness, cv2.LINE_AA)

plt.imshow(text_img[:,:,::-1])
plt.title('Apollo 11 with description- RGB')
plt.axis('off')
plt.show()
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```
rect_img = apollo.copy()
rect_img = cv2.rectangle(rect_img, (500,50), (800,625), (255, 0, 255), thickness = 10, lineType = cv2.LINE_8)
```

#### 13. Display the final annotated image.
```
plt.imshow(rect_img[:, :, ::-1])
plt.title('Apollo 11 with tower and rocket annotated - RGB')
plt.axis('off')
plt.show()
```

#### 14. Read the image ('Boy.jpg').
```
img = cv2.imread('boy.jpg', cv2.IMREAD_COLOR)
```

#### 15. Adjust the brightness of the image.
```
# Create a matrix of ones (with data type float64)
matrix = np.ones(img.shape, dtype = 'uint8') * 45
```

#### 16. Create brighter and darker images.
```
bright = cv2.add(img, matrix)
dark = cv2.subtract(img, matrix)
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
plt.figure(figsize = [31,11])
plt.subplot(131); plt.imshow(dark[:, :, ::-1]); plt.title('Darker')
plt.subplot(132); plt.imshow(img[:, :, ::-1]); plt.title('Original')
plt.subplot(133); plt.imshow(bright[:, :, ::-1]); plt.title('Brighter')
```

#### 18. Modify the image contrast.
```
matrix1 = np.ones(img.shape) * 0.85
matrix2 = np.ones(img.shape) * 1.28

low  = np.uint8(cv2.multiply(np.float64(img), matrix1))
high = np.uint8(cv2.multiply(np.float64(img), matrix2))
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
plt.figure(figsize = [18,5])
plt.subplot(131); plt.imshow(low[:, :, ::-1]); plt.title('Lower Contrast')
plt.subplot(132); plt.imshow(img[:, :, ::-1]); plt.title('Original')
plt.subplot(133); plt.imshow(high[:, :, ::-1]); plt.title('Higher Contrast')
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
b, g, r = cv2.split(img)

plt.figure(figsize = [20, 10])
plt.subplot(141); plt.imshow(r); plt.title('Red Channel')
plt.subplot(142); plt.imshow(g); plt.title('Green Channel')
plt.subplot(143); plt.imshow(b); plt.title('Blue Channel')

Merg = cv2.merge((r, g, b))

plt.subplot(144)
plt.imshow(Merg)
plt.title('Merged Output')
```

#### 21. Merged the R, G, B , displays along with the original image
```
plt.figure(figsize = [18,10])
plt.subplot(121); plt.imshow(Merg); plt.axis('off'); plt.title('Merged Output')
plt.subplot(122); plt.imshow(img[:,:,::-1]); plt.axis('off'); plt.title('Original Input')
```

#### 22. Split the image into the H, S, V components & Display the channels.
```
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

plt.figure(figsize = [18, 7])
plt.subplot(141); plt.imshow(h); plt.title('H Channel')
plt.subplot(142); plt.imshow(s); plt.title('S Channel')
plt.subplot(143); plt.imshow(v); plt.title('V Channel')

merg = cv2.merge([h,s,v])
merg_rgb = cv2.cvtColor(merg, cv2.COLOR_HSV2RGB)

plt.subplot(144); plt.imshow(merg_rgb); plt.title('Merged')
```

#### 23. Merged the H, S, V, displays along with original image.
```
plt.figure(figsize = [18,10])
plt.subplot(121); plt.imshow(merg_rgb); plt.axis('off'); plt.title('Merged Output')
plt.subplot(122); plt.imshow(img[:,:,::-1]); plt.axis('off'); plt.title('Original Input')
```

## Output:
- **i)** Read and Display an Image.  
- **ii)** Adjust Image Brightness.  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.

