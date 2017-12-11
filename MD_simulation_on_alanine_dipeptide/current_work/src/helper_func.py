from config import *
class Helper_func(object):
    def __init__(self):
        return

    @staticmethod
    def get_mutual_info_of_two_continuous_vars(temp_var_0, temp_var_1, bins=10, normalization=True):
        temp_hist_0, _ = np.histogramdd(temp_var_0, bins=bins)
        temp_hist_1, _ = np.histogramdd(temp_var_1, bins=bins)
        temp_hist_2, _ = np.histogramdd(np.array([temp_var_0, temp_var_1]).T, bins=bins)
        temp_hist_0 /= temp_hist_0.sum()
        temp_hist_1 /= temp_hist_1.sum()
        temp_hist_2 /= temp_hist_2.sum()
        result = np.sum([temp_hist_2[item_x, item_y] * np.log(
            temp_hist_2[item_x, item_y] / temp_hist_0[item_x] / temp_hist_1[item_y])
                     for item_x in range(bins) for item_y in range(bins) if temp_hist_2[item_x, item_y] != 0])
        if normalization:
            entropy_0 = - np.sum(temp_hist_0 * np.log(temp_hist_0))
            entropy_1 = - np.sum(temp_hist_1 * np.log(temp_hist_1))
            result /= (0.5 * (entropy_0 + entropy_1))
        return result
