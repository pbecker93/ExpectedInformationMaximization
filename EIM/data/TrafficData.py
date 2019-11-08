import os
from PIL import Image
import numpy as np
from numba import jit


class StanfordData:
    min_vals = np.array([[[3.5, 7.5]]])
    max_vals = np.array([[[1408.5, 1908.]]])

    def __init__(self, base_path, trajectory_length=5, seed=0):

        self._base_path = base_path
        self._rng = np.random.RandomState(seed=seed)
        self._trajectory_length = trajectory_length

        self.raw_data = self.read_data(os.path.join(self._base_path, "annotations.txt"))

        raw_train_samples = self.raw_data[:7500]
        self._shift = StanfordData.min_vals
        self._scale = StanfordData.max_vals - StanfordData.min_vals
        raw_train_samples = self.standardize_data(raw_train_samples)
        self.train_samples = np.reshape(raw_train_samples, [raw_train_samples.shape[0], -1])

        raw_val_samples = self.raw_data[7500:]
        raw_val_samples = self.standardize_data(raw_val_samples)
        self.val_samples = np.reshape(raw_val_samples, [raw_val_samples.shape[0], -1])
        self.test_samples = np.copy(self.val_samples)

        self._mask = self.get_mask()

    def standardize_data(self, data):
        return 2 * ((data - self._shift) / self._scale) - 1

    def de_standardize_data(self, data):
        return self._scale * (data + 1) / 2 + self._shift

    def read_data(self, annotations_path):
        traj_dict = {}
        type_dict = {}
        with open(annotations_path, "r") as f:

            for l in f:
                split_line = l.split(" ")
                if int(split_line[-2]) == 0:
                    id = int(split_line[0])
                    x = (int(split_line[1]) + int(split_line[3])) / 2
                    y = (int(split_line[2]) + int(split_line[4])) / 2
                    type = split_line[-1]
                    if id not in traj_dict.keys():
                        traj_dict[id] = [[x, y]]
                    else:
                        traj_dict[id].append([x, y])
                    if id not in type_dict.keys():
                        type_dict[id] = type

        traj_list = []

        for k in traj_dict.keys():
            traj = traj_dict[k]
            for i in range(0, len(traj) - self._trajectory_length, 1):
                t = np.stack([traj[i + j] for j in range(self._trajectory_length)], axis=0)
                traj_list.append(t)

        trajectories = np.stack(traj_list, 0)
        self._rng.shuffle(trajectories)
        print("Generated", trajectories.shape[0], "Trajectories")
        return trajectories

    def get_img(self):
        img = Image.open(os.path.join(self._base_path, "reference.jpg"))
        return np.asarray(img)

    def eval_fn(self, points):
        inval_points = 0
        for p in points:
            p_int = np.floor(p + 0.5).astype(np.int)
            if p_int[1] >= self._mask.shape[0] or p_int[1] < 0 or p_int[0] >= self._mask.shape[1] or p_int[0] < 0:
                inval_points += 1
            else:
                inval_points += self._mask[p_int[1], p_int[0]]
        return inval_points / points.shape[0]

    def get_mask(self):
        mask = dict(np.load(os.path.join(self._base_path, "mask.npz")))["mask"]
        return mask

    def get_overlay_img(self):
        overlay_image_as_array = \
            (255 * np.stack([self._mask, 1 - self._mask, np.zeros_like(self._mask)], axis=-1)).astype(np.uint8)
        overlay_image = Image.fromarray(overlay_image_as_array, mode="RGB")

        img_with_mask = Image.blend(Image.fromarray(self.get_img()), overlay_image, alpha=0.25)
        return np.array(img_with_mask)

    def get_add_feat(self, samples):
        samples = self.de_standardize_data(np.reshape(samples, [-1, self._trajectory_length, 2]))
        add_feat = self._get_add_feat(samples, self._mask, self._trajectory_length)
        return add_feat

    @staticmethod
    @jit(nopython=True)
    def _get_add_feat(samples, mask, trajectory_length):
        add_feat = np.zeros((samples.shape[0], trajectory_length))
        for j in range(samples.shape[0]):
            for i in range(5):
                p_int0 = int(samples[j, i, 0] + 0.5)
                p_int1 = int(samples[j, i, 1] + 0.5)
                if p_int1 >= mask.shape[0] or p_int1 < 0 or p_int0 >= mask.shape[1] or p_int0 < 0:
                    add_feat[j, i] = 1.0
                else:
                    add_feat[j, i] = mask[p_int1, p_int0]
        return add_feat


class LankershimData:

    min_vals = np.array([[[-100.481, 3.362]]])
    max_vals = np.array([[[ 94.76, 1659.48]]])

    def __init__(self, base_path, num_train_samples=10000, num_test_samples=10000, num_vals_samples=5000, seed=0):
        self._path = base_path
        self._trajectory_length = 5

        self._traj_raw = dict(np.load(os.path.join(self._path, "lankershim.npz")))["trajectories"]

        self._rng = np.random.RandomState(seed)
        self.standardized_trajs = self.standardize_data(self._traj_raw)

        rand_idx = self._rng.permutation(self.standardized_trajs.shape[0])
        train_idx = rand_idx[:num_train_samples]
        test_idx = rand_idx[num_train_samples:num_test_samples + num_train_samples]
        val_idx = rand_idx[num_train_samples + num_test_samples: num_train_samples + num_test_samples + num_vals_samples]

        self.train_samples = self.standardized_trajs[train_idx].reshape([-1, 10])
        self.test_samples = self.standardized_trajs[test_idx].reshape([-1, 10])
        self.val_samples = self.standardized_trajs[val_idx].reshape([-1, 10])

        self._mask = self.get_mask()

    def standardize_data(self, data):
        return 2 * ((data - LankershimData.min_vals) / (LankershimData.max_vals - LankershimData.min_vals)) - 1

    def de_standardize_data(self, data):
        return 1000 * data + 1000

    def get_overlay_img(self):
        return self.get_mask()

    def get_mask(self):
        mask = np.squeeze(dict(np.load(os.path.join(self._path, "mask2.npz")))["mask"]).astype(np.float32)
        return mask

    def get_mask_orig_scale(self):
        scale = LankershimData.max_vals - LankershimData.min_vals
        mask = Image.fromarray(self.get_mask(), mode="F")
        mask = mask.resize((scale[..., 0], scale[..., 1]))
        return np.array(mask)

    def eval_fn(self, points):
        val_points = 0
        for p in points:
            p_int = np.floor(p + 0.5).astype(np.int)
            if p_int[1] >= self._mask.shape[0] or p_int[1] < 0 or p_int[0] >= self._mask.shape[1] or p_int[0] < 0:
                val_points += 1
            else:
                val_points += self._mask[p_int[1], p_int[0]]
        return 1 - val_points / points.shape[0]

    def get_add_feat(self, samples):
        samples = self.de_standardize_data(np.reshape(samples, [-1, self._trajectory_length, 2]))
        add_feat = self._get_add_feat(samples, self._mask, self._trajectory_length)
        return add_feat

    @staticmethod
    @jit(nopython=True)
    def _get_add_feat(samples, mask, trajectory_length):
        add_feat = np.zeros((samples.shape[0], trajectory_length))
        for j in range(samples.shape[0]):
            for i in range(5):
                p_int0 = int(samples[j, i, 0] + 0.5)
                p_int1 = int(samples[j, i, 1] + 0.5)
                if p_int1 >= mask.shape[0] or p_int1 < 0 or p_int0 >= mask.shape[1] or p_int0 < 0:
                    add_feat[j, i] = 1.0
                else:
                    add_feat[j, i] = mask[p_int1, p_int0]
        return add_feat