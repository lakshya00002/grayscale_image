import numpy as np
from scipy.ndimage import convolve

class Image_2D:
    def __init__(self, width, height, initial_value):
        self.width = width
        self.height = height
        self.data = np.full((height, width), initial_value, dtype=int)
    
    def set_pixel(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y][x] = value
        else:
            raise ValueError("Pixel coordinates are out of bounds.")
    
    def get_pixel(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x]
        else:
            raise ValueError("Pixel coordinates are out of bounds.")
    
    def convolve(self, kernel):
        if len(kernel) % 2 == 0 or len(kernel[0]) % 2 == 0:
            raise ValueError("Kernel size should be odd.")
        new_data = convolve(self.data, kernel, mode='constant', cval=0.0)
        new_image = Image_2D(self.width, self.height, 0)
        new_image.data = new_data
        return new_image
    
    def adjust_brightness(self, adjustment_value):
        self.data = np.clip(self.data + adjustment_value, 0, 255)
    
    def adjust_contrast(self, contrast_factor):
        mean = np.mean(self.data)
        self.data = np.clip(contrast_factor * (self.data - mean) + mean, 0, 255)
    
    def apply_blur(self):
        blur_kernel = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]]) / 9
        self.data = convolve(self.data, blur_kernel, mode='constant', cval=0.0)
    
    def display(self):
        for row in self.data:
            print(' '.join(map(str, row)))

# Example usage:
width, height = 5, 5
initial_value = 100
image = Image_2D(width, height, initial_value)
image.set_pixel(2, 2, 255)
print("Original Image:")
image.display()

brightness_adjustment = 50
image.adjust_brightness(brightness_adjustment)
print("\nAfter Brightness Adjustment:")
image.display()

contrast_factor = 1.2
image.adjust_contrast(contrast_factor)
print("\nAfter Contrast Adjustment:")
image.display()

blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9
blurred_image = image.convolve(blur_kernel)
print("\nAfter Convolution with Blur Kernel:")
blurred_image.display()

image.apply_blur()
print("\nAfter Applying Blur:")
image.display()
