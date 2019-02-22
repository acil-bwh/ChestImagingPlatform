from cip_python.dcnn.data.h5_dataset import H5Dataset
import scipy.ndimage.interpolation as scipy_interpolation
import numpy as np

class dummy(object):
    def __init__(self):
        self.xs_sizes = ((64, 64, 64),(32,32,32))
        self.ys_sizes = ((57, 6), (57,6), (57,6))
    def format_data_to_network(self, xs, ys):
        xss = [None] * len(xs)
        #for i in range(len(xs)):
        # xss[0] = scipy_interpolation.zoom(xs[0],[0.5,0.5,0.5])
        # xss[1] = scipy_interpolation.zoom(xs[1], [0.25, 0.25, 0.25])
        xss[0] = xs[0][:64,:64,:64]
        xss[1] = xs[1][:32, :32, :32]
        return xss, ys
    def get_xs_ys_size(self):
        return self.xs_sizes, self.ys_sizes

network = dummy()


p = "/data/DNNData/jo780/datasets/3D/2018-01-23_COPDGene_withSHARP_trainval.h5"

ds = H5Dataset(p, batch_size=6,
               train_ixs=np.array([0, 1, 3, 4, 7, 10], np.int),
               validation_ixs=np.array([2,5]),
               test_ixs=np.array([12,13]),
               num_augmented_train_data_points_per_original_data_point=1,
               use_pregenerated_augmented_train_data=True,
               network=network,
               xs_dataset_names=('images', 'images'),
               ys_dataset_names=('labels', 'labels', 'labels'),
               pregenerated_augmented_xs_dataset_names=('images_augmented', 'images_augmented'),
               pregenerated_augmented_ys_dataset_names=('labels_augmented', 'labels_augmented', 'labels_augmented'),
               shuffle_training=False
               )
xs1, ys1 = ds.get_next_batch(12, 1)
xs, ys = ds.get_next_batch(2, 2)
xs, ys = ds.get_next_batch(2, 3)

# ds = H5Dataset(p, batch_size=6,
#                train_ixs=np.array([0, 1, 3, 4, 7, 10], np.int),
#                validation_ixs=np.array([2,5]),
#                test_ixs=np.array([12,13]),
#                num_augmented_train_data_points=2,
#                use_pregenerated_augmented_train_data=True,
#                network=network,
#                xs_dataset_names=('images', 'images'),
#                ys_dataset_names=('labels', 'labels', 'labels'),
#                pregenerated_augmented_xs_ds_names=('images_augmented', 'images_augmented'),
#                pregenerated_augmented_ys_ds_names=('labels_augmented', 'labels_augmented', 'labels_augmented'),
#                shuffle_training=False
#                )

# xs2, ys2 = ds.get_next_batch(6, 1)
# for i in range(2):
#     assert(np.array_equal(xs2[i], xs1[i]))
# pass

xs2, ys2 = ds.get_next_batch(10, 1)
# assert(np.array_equal(xs2[i], xs1[i]))
assert ds.current_epoch == 2, "{}".format(ds.current_epoch)
xs2, ys2 = ds.get_next_batch(3, 1)
assert ds.current_epoch == 3, "{}".format(ds.current_epoch)