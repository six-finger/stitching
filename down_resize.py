import cv2
from pathlib import Path
from matplotlib import pyplot as plt

def get_image_paths(img_dir, img_set):
    return [str(path.relative_to('.')) for path in Path(f'{img_dir}').rglob(f'{img_set}*')]
def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

origin_img_dir = 'imgs'
warped_img_dir = 'img_dir'
img_set = 'DSC00'

weihai_imgs = get_image_paths(warped_img_dir, img_set)

for img in weihai_imgs:
    image = cv2.imread(img)
    scale_percent = 0.2
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # plot_image(resized_image)
    save_path = img[:len(warped_img_dir)+1]+'small_'+img[len(warped_img_dir)+1:]
    cv2.imwrite(save_path, image)
    
weihai_imgs = get_image_paths(origin_img_dir, img_set)

for img in weihai_imgs:
    image = cv2.imread(img)
    scale_percent = 0.2
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # plot_image(resized_image)
    save_path = img[:len(origin_img_dir)+1]+'small_'+img[len(origin_img_dir)+1:]
    cv2.imwrite(save_path, image)
