import numpy as np
import nibabel as nib


def create_path(task_number=1, sub_number=101, data_path='./data',
                data_type='masked_T2', extension='.nii.gz'):
    """ Return path to fixed file
    :param task_number: number of task
    :param sub_number: number of data
    :param data_path: path to data
    :param data_type: type of data
    :param extension: extension of the file
    :return: path to file with data
    """
    return data_path + f"/Task{task_number}" + 2 * f"/sub-{sub_number}" + \
        "_space-T1_desc-" + data_type + extension


def load_data_as_array(task_number=1, sub_number=101, data_path='./data',
                       data_type='masked_T2', extension='.nii.gz'):
    """ Return fixed data as numpy.array
        :param task_number: number of task
        :param sub_number: number of data
        :param data_path: path to data
        :param data_type: type of data
        :param extension: extension of the file
        :return: data as numpy.array
        """
    return np.array(nib.load(create_path(task_number, sub_number, data_path,
                                         data_type, extension)).dataobj)
