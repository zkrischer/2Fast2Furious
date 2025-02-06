import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from machinevisiontoolbox import Image, colorspace_convert

def apply_hsv_threshold(img, hsv_values):
    h_min, s_min, v_min, h_max, s_max, v_max = hsv_values
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    hsv_image = colorspace_convert(img.image,'rgb','hsv')
    mask = ((hsv_image >= lower) & (hsv_image <= upper)).all(axis=2)
    result = hsv_image * mask[:, :, None]
    return result

def update(val, img, ax_img, fig, sliders):
    h_min = sliders['HMin'].val
    s_min = sliders['SMin'].val
    v_min = sliders['VMin'].val
    h_max = sliders['HMax'].val
    s_max = sliders['SMax'].val
    v_max = sliders['VMax'].val
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = ((img.image >= lower) & (img.image <= upper)).all(axis=2)
    result = img.image * mask[:, :, None]
    
    ax_img.set_data(result)
    fig.canvas.draw_idle()

def img_test(image):
    img = Image(image)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.4)
    hsv_image = colorspace_convert(img.image,'rgb','hsv')
    ax_img = ax.imshow(hsv_image)
    ax.set_title("HSV Thresholding")
    
    sliders = {}
    slider_positions = np.linspace(0.05, 0.35, 6)
    labels = ['HMin', 'SMin', 'VMin', 'HMax', 'SMax', 'VMax']
    ranges = [(0, 255), (0, 255), (0, 255), (0, 255), (0, 255), (0, 255)]
    
    for label, pos, (min_val, max_val) in zip(labels, slider_positions, ranges):
        ax_slider = plt.axes([0.25, pos, 0.65, 0.03])
        sliders[label] = Slider(ax_slider, label, min_val, max_val, valinit=max_val if 'Max' in label else min_val)
        sliders[label].on_changed(lambda val: update(val, img, ax_img, fig, sliders))
    
    plt.show()

def threshold_test():
    images = glob.glob('img/*.jpg')
    result = []
    for image_path in images: 
        img = Image.Read(image_path)
        # img_resized = img.resize((200, 200))
        result.append(img.image)
    
    if not result:
        print("No images found in img/*.png")
        return
    
    img_concat = np.hstack(result)
    img_test(img_concat[115:, :])


def imges():
    
    h_min = 136.5
    s_min = 0
    v_min = 0
    h_max = 255
    s_max = 72.5
    v_max = 85.5

    hsv_values = [0,0,0,244,244,244]#[h_max, s_max, v_max, h_min, s_min, v_min]

    images = glob.glob('img/*.jpg')

    for image_path in images:
        img = Image.Read(image_path)
        img_resized = img[:,115:]
        thresholded_img = apply_hsv_threshold(img_resized, hsv_values)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img_resized.image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        
        axs[1].imshow(thresholded_img)
        axs[1].set_title('Thresholded Image')
        axs[1].axis('off')
        
        plt.show()

def main():
    imges()
   

if __name__ == "__main__":
    main()
