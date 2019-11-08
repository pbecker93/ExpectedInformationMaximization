import numpy as np
from PIL import Image, ImageDraw
import scipy.interpolate as inter
from numba import jit
import tensorflow as tf


class ObstacleData:

    def __init__(self, num_train_samples, num_test_samples, num_val_samples=0, samples_per_context=10, seed=0,
                 num_obstacles=4):
        self._rng = np.random.RandomState(seed)
        self._x_res, self._y_res = 200, 100
        self._x_rad, self._y_rad = 5, 15
        self._base_x_points = 2 * (np.arange(1, num_obstacles + 1) / (num_obstacles + 1)) - 1
        self._num_obstacles = num_obstacles
        self._context_dim = 2 * num_obstacles
        self._sample_dim = 2 * num_obstacles #+ 2
        self._samples_per_context = samples_per_context

        self._pdf_c = tf.constant(- 0.5 * np.log(2 * np.pi), dtype=tf.float32)
        self._uu = 0.8
        self._ll = -0.8
        self._start_end_std = 5e-2
        self._x_std = 1e-2
        self._y_std = 5e-2

        assert num_train_samples % samples_per_context == 0
        assert num_test_samples % samples_per_context == 0
        assert num_val_samples % samples_per_context == 0 or num_val_samples < 1

        self.raw_train_samples = \
            self._sample_data_set(int(num_train_samples / samples_per_context), samples_per_context)
        self.raw_test_samples = self._sample_data_set(int(num_test_samples / samples_per_context), samples_per_context)
        if num_val_samples > 0:
            self.raw_val_samples = \
                self._sample_data_set(int(num_val_samples / samples_per_context), samples_per_context)
        else:
            self.raw_val_samples = None

        print("Train Reward:", self.rewards_from_contexts(self.raw_train_samples[0], self.raw_train_samples[1]))
        print("Test Reward:", self.rewards_from_contexts(self.raw_test_samples[0], self.raw_test_samples[1]))
        print("Validation Reward:", self.rewards_from_contexts(self.raw_val_samples[0], self.raw_val_samples[1]))
        self._build_data_sets(self.raw_train_samples, self.raw_test_samples, self.raw_val_samples)

    def _make_contexts_and_samples(self, raw_samples):
        contexts = np.tile(np.expand_dims(raw_samples[0], 1), [1, self._samples_per_context, 1])
        return np.reshape(contexts, [-1, self._context_dim]), np.reshape(raw_samples[1], [-1, self._sample_dim])

    def _build_data_sets(self, raw_train_samples, raw_test_samples, raw_val_samples):
        train_contexts, train_samples = self._make_contexts_and_samples(raw_train_samples)
        train_rand_idx = self._rng.permutation(train_contexts.shape[0])
        self.train_samples = train_contexts[train_rand_idx], train_samples[train_rand_idx]

        if raw_val_samples is not None:
            val_contexts, val_samples = self._make_contexts_and_samples(raw_val_samples)
            val_rand_idx = self._rng.permutation(val_contexts.shape[0])
            self.val_samples = val_contexts[val_rand_idx], val_samples[val_rand_idx]

        test_contexts, test_samples = self._make_contexts_and_samples(raw_test_samples)
        test_rand_idx = self._rng.permutation(test_contexts.shape[0])
        self.test_samples = test_contexts[test_rand_idx], test_samples[test_rand_idx]

    def _sample_data_set(self, num_contexts, samples_per_context):
        contexts = np.zeros([num_contexts, self._context_dim])

        for i in range(self._num_obstacles):
            contexts[:, 2 * i] = self._base_x_points[i] + self._rng.normal(loc=0, scale=0.04, size=num_contexts)
            contexts[:, 2 * i + 1] = self._rng.uniform(-0.5, 0.5, size=num_contexts)

        samples = np.zeros([num_contexts, samples_per_context, self._sample_dim])
        for i in range(num_contexts):
            sample_list = []
            while len(sample_list) < samples_per_context:
                x = self._create_sample(contexts[i])
                sample_list.append(x)
            samples[i] = np.array(sample_list)
        return contexts, samples

    def _create_sample(self, context):
        x = np.zeros(self._sample_dim)
       # x[0] = self._rng.normal(0, self._start_end_std)
        for i in range(0, self._num_obstacles):
            x[2 * i] = context[2 * i] + self._rng.normal(loc=0, scale=self._x_std)
            ul = context[2 * i + 1] + 0.25
            lu = context[2 * i + 1] - 0.25
            w_l = (lu - self._ll) / (self._uu - ul + lu - self._ll)
            m_l = self._ll + 0.5 * (lu - self._ll)
            m_u = ul + 0.5 * (self._uu - ul)
            eps = self._rng.uniform(0, 1)
            m = m_l if eps < w_l else m_u
            x[2 * i + 1] = self._rng.normal(m, self._y_std)
       # x[-1] = self._rng.normal(0, self._start_end_std)
        return x

    def img_from_context(self, context):
        img = Image.new('F', (self._x_res, self._y_res), 0.0)
        draw = ImageDraw.Draw(img)
        for i in range(self._num_obstacles):
            x, y = (context[2 * i] + 1) / 2, (context[2 * i + 1] + 1) / 2
            draw.ellipse((self._x_res * x - self._x_rad, self._y_res * y - self._y_rad,
                          self._x_res * x + self._x_rad, self._y_res * y + self._y_rad), fill=1.0)
        return np.asarray(img)

    def get_spline(self, x):
        x_ext = np.zeros(2 * self._num_obstacles + 4, dtype=x.dtype)
        x_ext[0] = 0.0
        x_ext[1] = 0.5
        x_ext[2:-2] = (x + 1) / 2
        x_ext[-2] = 1.0
        x_ext[-1] = 0.5 #'(x[-1] + 1) / 2
        k = "quadratic" if self._num_obstacles == 1 else "cubic"
        return inter.interp1d(x_ext[::2], x_ext[1::2], kind=k)
        #return inter.splrep(x_ext[::2], x_ext[1::2], k=k)

    def reward_from_context(self, context, x):
        return self.reward_from_img(self.img_from_context(context), x)

    def reward_from_img(self, img, x):
        spline = self.get_spline(x)
        x_eval = np.arange(0, 1, 1 / self._x_res)
        y_eval = spline(x_eval)
        eval_idx_u = np.ceil(self._y_res * y_eval).astype(np.int)
        eval_idx_l = np.floor(self._y_res * y_eval).astype(np.int)
        return - self._compute_loss(img, eval_idx_u, eval_idx_l, self._x_res, self._y_res)

    def rewards_from_contexts(self, context, x):
        rewards = np.zeros(x.shape[:2])
        for i in range(x.shape[0]):
            img = self.img_from_context(context[i])
            for j in range(x.shape[1]):
                rewards[i, j] = self.reward_from_img(img, x[i, j])
        return np.mean(rewards), np.count_nonzero(rewards) / np.prod(rewards.shape)

    #def get_reward_img(self, img, x, width=5, step=1):
    #    assert width % 2 == 1, "Currently only implemented for odd width values"
    #    rad = (width - 1) / 2
    #    patch_x_dim = int(self._x_res / step)
    #    spline = self.get_spline(x)
    #    x_eval = np.arange(0, 1, 1 / patch_x_dim)
    #    y_eval = spline(x_eval)
    #    upper_idx = np.ceil(self._y_res * y_eval + rad).astype(np.int32)
    #    lower_idx = np.floor(self._y_res * y_eval - rad).astype(np.int32)
    #    return self._get_patch(img, upper_idx, lower_idx, step, width)


    #def get_reward_imgs(self, contexts, x, width=5, step=1):
    #    patches = np.zeros([x.shape[0], x.shape[1], width, int(self._x_res / step)], dtype=np.float32)
    #    for i in range(x.shape[0]):
    #        img = self.img_from_context(contexts[i])
    ##        for j in range(x.shape[1]):
     #           patches[i, j] = self.get_reward_img(img, x[i, j], width=width, step=step)
     #   return patches

    #@staticmethod
    #@jit(nopython=True)
    #def _get_patch(img, upper_idx, lower_idx, step, patch_width):
    #    patch_x_dim = int(img.shape[1] / step)
    #    y_res = img.shape[0]
    #    patch = np.ones((patch_width, patch_x_dim))
    #    for i in range(patch_x_dim):
    #        img_idx = step * i
    #        # entire patch inside
    #        if upper_idx[i] < y_res and lower_idx[i] >= 0:
    #            patch[:, i] = img[lower_idx[i]: upper_idx[i], img_idx]
    #        # entire patch outside
    #        elif lower_idx[i] >= y_res or upper_idx[i] < 0:
    #            continue
    #        #border
    #        else:
    #            if lower_idx[i] < 0:
    #                patch[:, i] = np.concatenate((np.ones(0 - lower_idx[i]), img[:upper_idx[i], img_idx]))
    #            if upper_idx[i] >= y_res:
    #                patch[:, i] = np.concatenate((img[lower_idx[i]:, img_idx], np.ones(upper_idx[i] - y_res)))
    #    return patch


    @staticmethod
    @jit(nopython=True)
    def _compute_loss(img, y_eval_upper, y_eval_lower, x_dim, y_dim):
        loss = 0
        for i in range(x_dim):
            u_idx = y_eval_upper[i]
            l_idx = y_eval_lower[i]
            if l_idx < 0 or u_idx < 0 or l_idx >= y_dim or u_idx >= y_dim:
                loss += 1
            elif img[u_idx, i] > 0 or img[l_idx, i] > 0:
                loss += 1
        return loss

    def gauss_log_pdf(self, x, means, std):
        std += 1e-20
        sq_diff = (x - means) ** 2
        return self._pdf_c - tf.math.log(std) - .5 * sq_diff / (std ** 2)

    def ull(self, context, x):
        p = 0 #self.gauss_log_pdf(x[:, 0], 0, std=self._start_end_std)
        for i in range(self._num_obstacles):
            p += self.gauss_log_pdf(x[:, 2 * i], context[:, 2 * i], self._x_std)
            ul = context[:, 2 * i + 1] + 0.25
            lu = context[:, 2 * i + 1] - 0.25
            w_l = (lu - self._ll) / (self._uu - ul + lu - self._ll)
            w_u = (self._uu - lu) / (self._uu - ul + lu - self._ll)
            m_l = self._ll + 0.5 * (lu - self._ll)
            m_u = ul + 0.5 * (self._uu - ul)
            p += tf.math.log(w_l * tf.exp(self.gauss_log_pdf(x[:, 2 * i + 1], m_l, self._y_std)) +
                             w_u * tf.exp(self.gauss_log_pdf(x[:, 2 * i + 1], m_u, self._y_std)) + 1e-25)
        #p += self.gauss_log_pdf(x[:, -1], 0, std=self._start_end_std)
        return p

    @property
    def dim(self):
        return self._context_dim, self._sample_dim

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    #x_plt = np.arange(-3, 3, 0.01)

    data = ObstacleData(10000, 5000, 5000, num_obstacles=2)

    #patches = data.get_reward_imgs(data.raw_train_samples[0], data.raw_train_samples[1], width=5, step=4)


    x_plt = np.arange(0, 1, 0.01)
    for i in range(10):
        c = data.raw_train_samples[0][i]
        img = data.img_from_context(c)
        plt.figure()
        #plt.subplot(11, 1, 1)
        plt.imshow(img)
        #plt.scatter(c[::2], c[1::2])
        for j in range(10):
            spline = data.get_spline(data.raw_train_samples[1][i, j])
            y_plt = spline(x_plt)
            #plt.subplot(11, 1, 1)
            plt.plot(200 * x_plt, 100 * y_plt)
           # plt.subplot(11, 1, j+2)
          #  plt.imshow(patches[i, j])
    plt.show()








