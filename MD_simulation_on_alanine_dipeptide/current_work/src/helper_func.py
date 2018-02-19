from config import *
from scipy.special import erf

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

    @staticmethod
    def check_center_of_mass_is_at_origin(result):
        coords_of_center_of_mass_after = [[np.average(result[item, ::3]), np.average(result[item, 1::3]),
                                           np.average(result[item, 2::3])]
                                          for item in range(result.shape[0])]
        return np.all(np.abs(np.array(coords_of_center_of_mass_after).flatten()) < 1e-5)

    @staticmethod
    def remove_translation(coords):  # remove the translational degree of freedom
        if len(coords.shape) == 1:  # convert 1D array (when there is only one coord) to 2D array
            coords = coords.reshape((1, coords.shape[0]))
        number_of_atoms = coords.shape[1] / 3
        coords_of_center_of_mass = [[np.average(coords[item, ::3]), np.average(coords[item, 1::3]),
                                     np.average(coords[item, 2::3])] * number_of_atoms
                                    for item in range(coords.shape[0])]
        result = coords - np.array(coords_of_center_of_mass)
        assert Helper_func.check_center_of_mass_is_at_origin(result)
        return result

    @staticmethod
    def get_gyration_tensor_and_principal_moments(coords):
        coords = Helper_func.remove_translation(coords)
        temp_coords = coords.reshape(coords.shape[0], coords.shape[1] / 3, 3)
        gyration = np.zeros((coords.shape[0], 3, 3))
        for xx in range(3):
            for yy in range(3):
                gyration[:, xx, yy] = (temp_coords[:, :, xx] * temp_coords[:, :, yy]).mean(axis=-1)
        moments_gyration = np.linalg.eig(gyration)[0]
        moments_gyration.sort(axis=-1)
        return gyration, moments_gyration[:, ::-1]

    @staticmethod
    def get_norm_factor(rcut, sig):
        rcut2 = rcut*rcut
        sig2 = 2.0*sig*sig
        normconst = np.sqrt( np.pi * sig2 ) * erf( rcut / (sqrt(2.0)*sig) ) - 2*rcut*exp( - rcut2 / sig2 )
        preerf = np.sqrt( 0.5 * np.pi * sig * sig ) / normconst
        prelinear = exp( - rcut2 / sig2 ) / normconst
        return normconst, preerf, prelinear

    @staticmethod
    def get_coarse_grained_count(dis, r_hi, rcut, sig):
        # TODO: test if this function is correct
        normconst, preerf, prelinear = Helper_func.get_norm_factor(rcut, sig)
        hiMinus = r_hi - rcut
        hiPlus = r_hi + rcut
        count = np.float64((dis < hiPlus).sum(axis=-1))
        temp_in_boundary_region = ((dis > hiMinus) & (dis < hiPlus))
        temp_correction = ( 0.5 + preerf * erf( np.sqrt(0.5) * (dis - r_hi)/sig ) \
                                             - prelinear * (dis - r_hi))
        # print count.shape, temp_in_boundary_region.shape, temp_correction.shape
        count -= (temp_in_boundary_region * temp_correction).sum(axis=-1)
        return count
