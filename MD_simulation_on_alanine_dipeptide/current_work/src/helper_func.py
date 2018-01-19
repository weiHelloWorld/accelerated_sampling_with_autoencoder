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

    @staticmethod
    def generate_alkane_residue_code_in_openmm_xml(num, name):
        print '''<Residue name="%s">
<Atom charge="0.09" name="H11" type="HGA3"/>
<Atom charge="0.09" name="H12" type="HGA3"/>
<Atom charge="0.09" name="H13" type="HGA3"/>
<Atom charge="-0.27" name="C1" type="CG331"/>''' % name
        for item in range(num - 2):
            print '''<Atom charge="0.09" name="H%d1" type="HGA2"/>
<Atom charge="0.09" name="H%d2" type="HGA2"/>
<Atom charge="-0.18" name="C%d" type="CG321"/>''' % (item + 2, item + 2, item + 2)
        print """<Atom charge="0.09" name="H%d1" type="HGA3"/>
<Atom charge="0.09" name="H%d2" type="HGA3"/>
<Atom charge="0.09" name="H%d3" type="HGA3"/>
<Atom charge="-0.27" name="C%d" type="CG331"/>
<Bond atomName1="H11" atomName2="C1"/>
<Bond atomName1="H12" atomName2="C1"/>
<Bond atomName1="H13" atomName2="C1"/>""" % (num, num, num, num)
        for item in range(num - 1):
            print """<Bond atomName1="C%d" atomName2="C%d"/>
<Bond atomName1="H%d1" atomName2="C%d"/>
<Bond atomName1="H%d2" atomName2="C%d"/>""" % (item + 1, item + 2, item + 2, item + 2, item + 2, item + 2)
        print """<Bond atomName1="H%d3" atomName2="C%d"/>
<AllowPatch name="MET1"/>
<AllowPatch name="MET2"/>
</Residue>""" % (num, num)
        return
    