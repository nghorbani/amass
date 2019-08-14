import numpy as np

def show_image(img_ndarray):
    '''
    Visualize images resulted from calling vis_smpl_params in Jupyternotebook
    :param img_ndarray: Nx400x400x3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    # fig.canvas.draw()
    # return True