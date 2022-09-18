import numpy as np
import time
import matplotlib.pyplot as plt


def _show_as_film_x(image, start, end, step=1, sleep_time=1):
    for i in np.arange(start, end, step):
        plt.imshow(image[i, :, :])
        plt.title(f'high: {i}')
        plt.show(block=False)
        time.sleep(sleep_time) if sleep_time != 0 else None
        plt.close()


def _show_as_film_y(image, start, end, step=1, sleep_time=1):
    for i in np.arange(start, end, step):
        plt.imshow(image[:, i, :])
        plt.title(f'high: {i}')
        plt.show(block=False)
        time.sleep(sleep_time) if sleep_time != 0 else None
        plt.close()


def _show_as_film_z(image, start, end, step=1, sleep_time=1):
    for i in np.arange(start, end, step):
        plt.imshow(image[:, :, i])
        plt.title(f'high: {i}')
        plt.show(block=False)
        time.sleep(sleep_time) if sleep_time != 0 else None
        plt.close()


def show_as_film(image, start=0, end=180, step=1, axis='x', sleep_time=1):
    """ show slices of 3D image one by one.
    !!! USE ONLY IN PYTHON CONSOLE !!!
    ========
    EXAMPLE:
        import load_data
        import data_viewer
        img = load_data.load_data_as_array()
        data_viewer.show_as_film(img, start=30, end=150, step=20)
    ========
    :param image: 3D image as numpy.array
    :param start: int, first slice
    :param end: int, last slice
    :param step: int, difference between number of successive printed slices
    :param axis: the axis along which the next slices will be displayed
    :param sleep_time: time of displayed slices
    :return: None
    """
    globals()['_show_as_film_' + axis](image, start, end, step, sleep_time)
