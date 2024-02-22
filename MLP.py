from dataReader import getData
import matplotlib.pyplot as plt


if __name__ == "__main__":

    labels, images = getData()
    
    for i in range(9):
        plt.subplot(330+1+i)
        plt.imshow(images[i,:,:], cmap='gray')
    plt.show()
    
