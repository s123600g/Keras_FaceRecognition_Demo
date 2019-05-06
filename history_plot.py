from keras.models import Model
import FaceID
import matplotlib.pyplot as plt

'''
http://noahsnail.com/2017/04/29/2017-4-29-matplotlib%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%94%A8%E6%B3%95(%E5%9B%9B)%E2%80%94%E2%80%94%E8%AE%BE%E7%BD%AElegend%E5%9B%BE%E4%BE%8B/
https://blog.csdn.net/Quincuntial/article/details/70947363
https://www.jianshu.com/p/91eb0d616adb
'''
       

def plot_figure(history,dir,figure_classname):

    __plot_acc(history,dir,figure_classname)

    #plt.subplot(1,2,1)

    __plot_loss(history,dir,figure_classname)

    #plt.show()


def __plot_acc(hist,dir,figure_classname):
        
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
        
    # img size
    plt.figure(figsize=(8,8))
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title(figure_classname+'_Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train_acc', 'valid_acc'], loc='best')
    plt.tight_layout()
    plt.savefig(dir+figure_classname+'_Accuracy.png')
    #plt.show()

def __plot_loss(hist,dir,figure_classname):

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # img size
    plt.figure(figsize=(8,8))        
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(figure_classname+'_Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train_loss', 'valid_loss'], loc='best')
    plt.tight_layout()
    plt.savefig(dir+figure_classname+'_Loss.png')
    #plt.show()
        

