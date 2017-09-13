import stl10_input
import load_BSDS300
import load_images 

class DataHandler():
    """
    Load + Preprocess + handle batches
    """

    def __init__(self, name):
        self.dataset_name = name


    def load_data(self, grayscale, resize):
        if self.dataset_name == 'stl10':
            return stl10_input.load_data(grayscale=grayscale)
        elif self.dataset_name == 'BSDS3':
            pass
        else:
            return load_images.load(self.dataset_name, grayscale)

    def preprocess_data(self, grayscal, resize):
        pass

    def nextbatch(X, Y, batch_size=500, run_once=False):
        offset = 0
        data_len = X.shape[0]

        batch_Y = None # label
        batch_X = None # input image maybe with noise

        while True:
            if offset + batch_size <= data_len:
                batch_X = X[offset: batch_size + offset]
                batch_Y = Y[offset: batch_size + offset]
                offset = offset + batch_size
            else:
                if run_once:
                    raise StopIteration()
                batch_X = np.concatenate((X[offset: data_len], X[:batch_size - (data_len - offset)]), axis=0)
                batch_Y = np.concatenate((Y[offset: data_len], Y[:batch_size - (data_len - offset)]), axis=0)
                offset = batch_size - (data_len - offset)
            yield batch_X, batch_Y    
