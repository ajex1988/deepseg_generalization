import os
import config

exp_folder = '/home/data/liver_segmentation/systematic_study/experiments'

def batch_experiment():
    batch_size_list = [8, 16, 32, 64] # batch size candidates used in the exp
    repeat_time = 1 # repeat each experiment several times for more reliable results

    exp_task_folder = os.path.join(exp_folder,'batch_size')  # Save the model check points
    if not(os.path.exists(exp_task_folder)):
        os.makedirs(exp_task_folder)

    dataset_name = 'synthesized'
    exp_task_dataset_folder = os.path.join(exp_task_folder,dataset_name)
    if not os.path.exists(exp_task_dataset_folder):
        os.makedirs(exp_task_dataset_folder)

    for batch_size in batch_size_list:
        # create a folder for each batch size
        exp_task_dataset_batch_folder = os.path.join(exp_task_dataset_folder,'{0}'.format(batch_size))
        if not os.path.exists(exp_task_dataset_batch_folder):
            os.makedirs(exp_task_dataset_batch_folder)

        for r_idx in range(repeat_time):
            # creat a folder for each repeat
            exp_task_dataset_batch_repeat_folder = os.path.join(exp_task_dataset_batch_folder,'{0}'.format(r_idx))
            if not os.path.exists(exp_task_dataset_batch_repeat_folder):
                os.makedirs(exp_task_dataset_batch_repeat_folder)
    batch_configs = config.BatchSizeExpConfig()