
import numpy as np
import matplotlib.pylab as plt

#display original image, predicted mask and ground-truth.
def plot_prediction(image, predMask, gtMask=None, 
                    sup_title= 'result', save_path=None, auto_close=False):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3)
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(image)
    ax[1].imshow(predMask)
    if gtMask is not None:
        ax[2].imshow(gtMask)
        
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Ground-truth")
    ax[2].set_title("Prediction")
   
    figure.suptitle(sup_title)#, fontsize=30)
    
    # set the layout of the figure and display it
    #figure.tight_layout()
    #figure.show()
    
    if save_path is not None:
       plt.savefig(save_path, dpi=300)
    
    if auto_close:       
        plt.close()       
       
       
# display multipl images
#imgs: list of images (numpy format (H,W,3) or (H,W))
#titles: list of titles for the images above.
def plot_images(imgs, titles, sup_title=None, save_path=None, auto_close=False):
    nimgs = len(imgs)
    nrows = np.int32(np.round(np.sqrt(nimgs)))
    ncols = np.int32(np.ceil(nimgs/nrows))
    # initialize our figure
    figure, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    
    n = 0
    for i in range(nrows):
        for j in range(ncols):        
            if n<nimgs:
                ax[i, j].imshow(imgs[n])    
                ax[i, j].set_title(titles[n])
            ax[i, j].axis('off')
            n += 1
    
    if sup_title is not None:
        figure.suptitle(sup_title)#, fontsize=30)    
        
    figure.show()
    
    if save_path is not None:
       plt.savefig(save_path, dpi=300)       
       
    if auto_close:       
        plt.close()       
           