


class DefaultConfig():
    def __init__(self):
        pass
    def set_checkpoint_path(self):
        # Virtual method. Sub-class should override this.
        raise NotImplementedError()

    # Default values of training-related parameters
    resize_height = 256
    resize_width = 256

    batch_size = 16

class BatchSizeExpConfig(DefaultConfig):
    def __init__(self,exp_folder, dataset_name, batch_size_list):
        self.exp_folder = exp_folder
        self.dataset_name = dataset_name
        self.batch_size_list = batch_size_list
