import numpy as np
import sys
import warnings
import argparse

from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser


class FractalDimensionPhenotypes(Phenotypes):
    """
    General purpose class for generating fractal dimension phenotype.
    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:
    FDbc': fractal_dimension computed by box-counting

    Parameters
    ----------
    chest_regions : list of strings
        List of chest regions over which to compute the phenotypes.

    chest_types : list of strings
        List of chest types over which to compute the phenotypes.

    pairs : list of lists of strings
        Each element of the list is a two element list indicating a region-type
        pair for which to compute the phenotypes. The first entry indicates the
        chest region of the pair, and second entry indicates the chest type of
        the pair.

    pheno_names : list of strings, optional
        Names of phenotypes to compute. These names must conform to the
        accepted phenotype names, listed above. If none are given, all
        will be computed.

    """

    def __init__(self, chest_regions=None, chest_types=None, pairs=None,
                 pheno_names=None):
        c = ChestConventions()

        self.chest_regions_ = None
        if chest_regions is not None:
            tmp = []
            for m in range(0, len(chest_regions)):
                tmp.append(c.GetChestRegionValueFromName(chest_regions[m]))
            self.chest_regions_ = np.array(tmp)

        self.chest_types_ = None
        if chest_types is not None:
            tmp = []
            for m in range(0, len(chest_types)):
                tmp.append(c.GetChestTypeValueFromName(chest_types[m]))
            self.chest_types_ = np.array(tmp)

        self.pairs_ = None
        if pairs is not None:
            self.pairs_ = np.zeros([len(pairs), 2])
            inc = 0
            for p in pairs:
                assert len(p) % 2 == 0, \
                    "Specified region-type pairs not understood"
                r = c.GetChestRegionValueFromName(p[0])
                t = c.GetChestTypeValueFromName(p[1])
                self.pairs_[inc, 0] = r
                self.pairs_[inc, 1] = t
                inc += 1

        self.requested_pheno_names = pheno_names

        Phenotypes.__init__(self)

    def declare_pheno_names(self):  # TODO: modify this function to use ChestConvention names once tested

        """Creates the names of the phenotypes to compute

          Returns
          -------
          names : list of strings
          Phenotype names
        """
        cols = []
        cols.append('FDbc')
        return cols

    def get_cid(self):
        """Get the case ID (CID)
        """
        return self.cid_

    def execute(self, lm, cid, spacing, stencil=None, num_offsets=1, chest_regions=None, chest_types=None, pairs=None, pheno_names=None):
        """Compute the phenotypes for the specified structures.

        The following values are computed for the specified structures.
        'FDbc': fractal_dimension computed by box-counting

        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume

        stencil : array, shape ( X, Y, Z )
            The 3D stencil image array

        num_offsets : int
            Number of offsets to use. If set to 1, no offsets are generated. If greater than 1, then generate evenly
            spaced offsets along EACH axis and use the lowest box count for each scale. Please note that time
            increases are exponential!

        chest_regions : array, shape ( R ), optional
            Array of integers, with each element in the interval [0, 255],
            indicating the chest regions over which to compute the LAA. If none
            specified, the chest regions specified in the class constructor
            will be used. If chest regions, chest types, and chest pairs are
            left unspecified both here and in the constructor, then the
            complete set of entities found in the label map will be used.

        chest_types : array, shape ( T ), optional
            Array of integers, with each element in the interval [0, 255],
            indicating the chest types over which to compute the LAA. If none
            specified, the chest types specified in the class constructor
            will be used. If chest regions, chest types, and chest pairs are
            left unspecified both here and in the constructor, then the
            complete set of entities found in the label map will be used.

        pairs : array, shape ( P, 2 ), optional
            Array of chest-region chest-type pairs over which to compute the
            LAA. The first column indicates the chest region, and the second
            column indicates the chest type. Each element should be in the
            interal [0, 255]. If none specified, the pairs specified in the
            class constructor will be used. If chest regions, chest types, and
            chest pairs are left unspecified both here and in the constructor,
            then the complete set of entities found in the label map will be
            used.

        pheno_names : list of strings, optional
            Names of phenotypes to compute. These names must conform to the
            accepted phenotype names, listed above. If none are given, all
            will be computed. Specified names given here will take precedence
            over any specified in the constructor.

        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and chest region
            chest type phenotype quantities.
        """
        c = ChestConventions()
        if stencil is not None:
            assert len(stencil.shape) == len(lm.shape), \
                "Stencil image and label map are not the same dimension"

        dim = len(lm.shape)
        if stencil is not None:
            for i in range(0, dim):
                assert stencil.shape[0] == lm.shape[0], \
                    "Disagreement in Stencil and label map dimension"

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid
        self._spacing = spacing

        # Derive phenos to compute from list provided
        # As default use the list that is provided in the constructor of phenotypes
        phenos_to_compute = self.pheno_names_

        if pheno_names is not None:
            phenos_to_compute = pheno_names
        elif self.requested_pheno_names is not None:
            phenos_to_compute = self.requested_pheno_names

        # Deal with wildcard: default to main pheno list if wildcard is provided
        if c.GetChestWildCardName() in phenos_to_compute:
            phenos_to_compute = self.pheno_names_

        # Check validity of phenotypes
        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
                "Invalid phenotype name " + pheno_name

        rs = np.array([])
        ts = np.array([])
        ps = np.array([])
        if chest_regions is not None:
            rs = chest_regions
        elif self.chest_regions_ is not None:
            rs = self.chest_regions_
        if chest_types is not None:
            ts = chest_types
        elif self.chest_types_ is not None:
            ts = self.chest_types_
        if pairs is not None:
            ps = pairs
        elif self.pairs_ is not None:
            ps = self.pairs_

        parser = RegionTypeParser(lm)

        if rs.size == 0 and ts.size == 0 and ps.size == 0:
            rs = parser.get_all_chest_regions()
            ts = parser.get_chest_types()
            ps = parser.get_all_pairs()

        # Now compute the phenotypes and populate the data frame
        for r in rs:
            if r != 0:
                mask_region = parser.get_mask(chest_region=r)
                self.add_pheno_group(stencil, num_offsets, mask_region, c.GetChestRegionName(r),
                                     c.GetChestWildCardName(), phenos_to_compute)
        for t in ts:
            if t != 0:
                mask_type = parser.get_mask(chest_type=t)
                self.add_pheno_group(stencil, num_offsets, mask_type, c.GetChestWildCardName(),
                                     c.GetChestTypeName(t), phenos_to_compute)
        if ps.size > 0:
            for i in range(0, ps.shape[0]):
                if not (ps[i, 0] == 0 and ps[i, 1] == 0):
                    mask = parser.get_mask(chest_region=int(ps[i, 0]),
                                           chest_type=int(ps[i, 1]))
                    self.add_pheno_group(stencil, num_offsets, mask, c.GetChestRegionName(int(ps[i, 0])),
                                         c.GetChestTypeName(int(ps[i, 1])), phenos_to_compute)

        return self._df

    def add_pheno_group(self, stencil, num_offsets, mask, chest_region,
                        chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method.

        Parameters
        ----------
        stencil : array, shape ( X, Y, Z )
            The 3D stencil image array

        mask : boolean array, shape ( X, Y, Z ), optional
            Boolean mask where True values indicate presence of the structure
            of interest

        mask_region: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding region

        mask_type: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding type

        chest_region : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        chest_type : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        phenos_to_compute : list of strings
            Names of the phenotype used to populate the dataframe

        References
        ----------
        1. Schneider et al, 'Correlation between CT numbers and tissue
        parameters needed for Monte Carlo simulations of clinical dose
        distributions'
        """
        mask_sum = np.sum(mask)
        print("Computing phenos for region %s and type %s" % (chest_region, chest_type))
        if stencil is not None:
            stencil_mask = np.zeros(stencil.shape, dtype=stencil.dtype)
            # stencil_mask = stencil[mask]
            idx = np.nonzero(mask)
            stencil_mask[idx] = 1

            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                    "Invalid phenotype name " + pheno_name
                pheno_val = self.compute_FD_box_counting(stencil_mask, num_offsets=num_offsets)

                if pheno_val is not None:
                    self.add_pheno([chest_region, chest_type],
                                   pheno_name, pheno_val)

    def compute_FD_box_counting(self, arr, threshold=1.0, num_offsets=1, do_plot=False):
        """ This function takes as input a 1, 2, or 3D array and returns the fractal dimension as computed by box counting.

        Parameters
        ----------
        arr : array, shape (N, N)
            1 or 2 or 3D array containing raw pixel values to be analyzed.

        threshold : float
            Threshold value to binarize arr. Values above or equal to threshold are analyzed.

        do_plot : Boolean
            If true, displays a figure containing the binarized image and the loglog plot of scale vs box count

        num_offsets : int
            Number of offsets to use. If set to 1, no offsets are generated. If greater than 1, then generate evenly spaced
            offsets along EACH axis and use the lowest box count for each scale. Please note that time increases are
            exponential!

        Returns
        -------
        fractal_dim : float
            Corresponding fractal dimension of the arr

        """

        # TODO: arg checks

        # Convert to binary
        arr_bin = arr >= threshold

        # Minimum dimension
        dimension_min = min(arr_bin.shape)

        # WARNING: Imprecision if dimension_min is low
        if dimension_min < 2**6:
            warnings.warn('Possible Imprecision due to Low Dimenion!')

        # Number of generations based on greatest power of 2 less than minimum dimension
        n = 2**np.floor(np.log(dimension_min) / np.log(2))
        n = int(np.log(n) / np.log(2))

        # Box sizes to count (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 0, -1)

        # print(' Size, Box Count:')
        counts = []
        for size in sizes:
            # Iterate through offsets to find the lowest box counts
            bc_best = sys.maxsize

            # Create offset coordinates based on number of dimensions of the array
            if num_offsets >= size or num_offsets == -1:
                # Use all possible offset coordinates
                offset_coordinates = np.arange(size)
            else:
                # Use linspace to create the offset coordinates
                offset_coordinates = np.linspace(0, size, num=num_offsets, endpoint=False)
                offset_coordinates = [int(num) for num in offset_coordinates]

            # Create all combinations of offset_coordinates for dimensions 1 ~ 3
            offsets = []
            if arr.ndim == 1:
                offsets = [offset_coordinates]
            elif arr.ndim == 2:
                for offset_x in offset_coordinates:
                    for offset_y in offset_coordinates:
                        offsets.append([[offset_x, 0], [offset_y, 0]])
            elif arr.ndim == 3:
                for offset_x in offset_coordinates:
                    for offset_y in offset_coordinates:
                        for offset_z in offset_coordinates:
                            offsets.append([[offset_x, 0], [offset_y, 0], [offset_z, 0]])

            for offset in offsets:
                # Create offset iterations via padding
                arr_padded = np.pad(arr_bin, offset)

                # Box counting!
                bc = self.box_count(arr_padded, size)
                if bc < bc_best:
                    bc_best = bc
                    print('Size:', size, '/ Offset:', offset, '/ BC:', bc)

            counts.append(bc_best)

        # Remove entries with 0 boxes (Removes log(0) error)
        counts = np.array(counts)
        sizes = sizes[counts.nonzero()]
        counts = counts[counts.nonzero()]

        # # Adds bc for size = 1
        # sizes = np.append(sizes, 1)
        # counts = np.append(counts, np.count_nonzero(arr_bin))
        # print('Size: 1 / BC:', counts[-1])

        # Fit the successive log(sizes) with log(counts)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

        # # Visualize
        # if do_plot:
        #     fig = plt.figure()
        #
        #     if arr.ndim == 2:
        #         plt.subplot(221)
        #         plt.imshow(arr, aspect='equal')
        #         plt.title('Original')
        #
        #         plt.subplot(223)
        #         plt.imshow(arr_bin, aspect='equal', cmap=plt.gray())
        #         plt.title('Binary')
        #
        #         plt.subplot(122)
        #         plt.plot(np.log(sizes), np.log(counts), label='Raw')
        #         plt.plot(np.log(sizes), np.polyval(coeffs, np.log(sizes)), label='Fit')
        #         plt.xlabel('Sizes (log10)')
        #         plt.ylabel('Box Counts (log10)')
        #         plt.title('Box Counts and Fit (' + 'th:{0:.3f}, fd:{1:.3f}'.format(threshold, -coeffs[0]) + ')')
        #         plt.legend()
        #
        #         plt.tight_layout()
        #         plt.show()
        #
        #     if arr.ndim == 3:
        #         ax = fig.add_subplot(121, projection='3d')
        #         z, x, y = arr_bin.nonzero()
        #         ax.scatter(x, y, z, marker='s')
        #         plt.title('Binary')
        #
        #         plt.subplot(122)
        #         plt.plot(np.log(sizes), np.log(counts), label='Raw')
        #         plt.plot(np.log(sizes), np.polyval(coeffs, np.log(sizes)), label='Fit')
        #         plt.xlabel('Sizes (log10)')
        #         plt.ylabel('Box Counts (log10)')
        #         plt.title('Box Counts and Fit (' + 'th:{0:.3f}, fd:{1:.3f}'.format(threshold, -coeffs[0]) + ')')
        #         plt.legend()
        #
        #         plt.tight_layout()
        #         plt.show()

        return -coeffs[0]

    @staticmethod
    def box_count(z, k):
        s = z
        for i in range(z.ndim):
            s = np.add.reduceat(s, np.arange(0, z.shape[i], k), axis=i)

        # Count non-empty boxes (k**<dimension>)
        return len(np.where(s > 0)[0])

    # ---------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ TEST FUNCTIONS -----------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    # Generates test 2d fractal with fd = 1.694
    # Pseudo Sierpinski's triangle
    @staticmethod
    def generate_test_2d(n):
        assert(n > 3)
        size = 2**n
        first_row = np.zeros(size, dtype=int)
        first_row[int(size / 2) - 1] = 1
        rows = np.zeros((int(size / 2), size), dtype=int)
        rows[0] = first_row
        for i in range(1, int(size / 2)):
            rows[i] = (np.roll(rows[i - 1], -1) + rows[i - 1] + np.roll(rows[i - 1], 1)) % 2
        m = int(np.log(size) / np.log(2))
        return rows[0:2 ** (m - 1), 0:2 ** m]

    # Generates a 3d test fractal with fd = 2.726
    # Menger sponge
    @staticmethod
    def generate_test_3d_menger(n, g=-1):
        if g > n or g < 0:
            g = n
        size = 3**n
        arr = np.ones((size, size, size))
        for i in range(g):
            hole_coordinates = []
            for j in range(3**i):
                i_start = (3**(n-i-1)) + j * (3**(n-i))
                i_end = (3**(n-i-1) + 3**(n-i-1) - 1) + j * (3**(n-i))
                hole_coordinates.append((i_start, i_end))

            for x_coord in hole_coordinates:
                for y_coord in hole_coordinates:
                    arr[:, x_coord[0]:x_coord[1]+1, y_coord[0]:y_coord[1]+1] = 0
                    arr[x_coord[0]:x_coord[1]+1, :, y_coord[0]:y_coord[1]+1] = 0
                    arr[x_coord[0]:x_coord[1]+1, y_coord[0]:y_coord[1]+1, :] = 0

        return arr

    # Generates a 3d test fractal with fd = 1.892
    # Cantor dust
    @staticmethod
    def generate_test_3d_cantor(n, g=-1):
        if g > n or g < 0:
            g = n
        size = 3 ** n
        arr = np.ones((size, size, size))
        for i in range(g):
            for j in range(3 ** i):
                i_start = (3 ** (n - i - 1)) + j * (3 ** (n - i))
                i_end = (3 ** (n - i - 1) + 3 ** (n - i - 1) - 1) + j * (3 ** (n - i))
                arr[:, :, i_start:i_end + 1] = 0
                arr[:, i_start:i_end + 1, :] = 0
                arr[i_start:i_end + 1, :, :] = 0

        return arr


# Generates a 3d test fractal with fd = 1.892
# Cantor dust
def generate_test_3d_cantor(n, g=-1):
    if g > n or g < 0:
        g = n
    size = 3**n
    arr = np.ones((size, size, size))
    for i in range(g):
        for j in range(3**i):
            i_start = (3**(n - i - 1)) + j * (3**(n - i))
            i_end = (3**(n - i - 1) + 3**(n - i - 1) - 1) + j * (3**(n - i))
            arr[:, :, i_start:i_end+1] = 0
            arr[:, i_start:i_end+1, :] = 0
            arr[i_start:i_end+1, :, :] = 0

    return arr



if __name__ == "__main__":
    desc = """Generate Fractal Dimension (FD) phenotypes given input stencil file and segmentation data. \ 
              Specified chest-regions, chest-types, and region-type pairs indicate the parts of the image over which \ 
              the requested phenotypes are computed."""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--in_stencil', '-in_stencil', required=True,
                        help='Stencil file (.nrrd) generated from airway/vessel particles')
    parser.add_argument('--in_lm', '-in_lm', required=True,
                        help='Input label map containing structures of interest')
    parser.add_argument('--out_csv', '-out_csv', required=True,
                        help='Output csv file in which to store the computed dataframe')
    parser.add_argument('--cid', '-cid', required=True, help='Case id')
    parser.add_argument('--num_offsets', '-num_offsets',
                        help='Number of offsets to use for fractal dimension computation. \
                              If set to 1 (default value), no offsets are generated. If greater than 1, \
                              then generate evenly spaced offsets along EACH axis and use \
                              the lowest box count for each scale. Please note that time \
                              increases are exponential!', default=1)
    # Keep these arguments as strings for compatibility purposes
    parser.add_argument('-r', dest='chest_regions',
                        help='Chest regions. Should be specified as a \
                              common-separated list of string values indicating the \
                              desired regions over which to compute the phenotypes. \
                              E.g. LeftLung,RightLung would be a valid input.')
    parser.add_argument('-t', dest='chest_types',
                        help='Chest types. Scd phhould be specified as a \
                              common-separated list of string values indicating the \
                              desired types over which to compute the phenotypes. \
                              E.g.: Vessel,NormalParenchyma would be a valid input.')
    parser.add_argument('-p', dest='pairs',
                        help='Chest region-type pairs. Should be \
                              specified as a common-separated list of string values \
                              indicating what region-type pairs over which to compute \
                              phenotypes. For a given pair, the first entry is \
                              interpreted as the chest region, and the second entry \
                              is interpreted as the chest type. E.g. LeftLung,Vessel \
                              would be a valid entry. Note that region-type pairs are \
                              interpreted differently in the case of the TypeFrac \
                              phenotype. In that case, the specified pair is used to \
                              compute the fraction of the chest type within the \
                              chest region.')

    args = parser.parse_args()

    image_io = ImageReaderWriter()

    regions = None
    if args.chest_regions is not None:
        regions = args.chest_regions.split(',')
    types = None
    if args.chest_types is not None:
        types = args.chest_types.split(',')

    lm, lm_header = image_io.read_in_numpy(args.in_lm)
    stencil, stencil_header = image_io.read_in_numpy(args.in_stencil)

    spacing = lm_header['spacing']

    pairs = None
    if args.pairs is not None:
        tmp = args.pairs.split(',')
        assert len(tmp) % 2 == 0, 'Specified pairs not understood'
        pairs = []
        for i in range(0, len(tmp) // 2):
            pairs.append([tmp[2 * i], tmp[2 * i + 1]])

    paren_pheno = FractalDimensionPhenotypes(chest_regions=regions, chest_types=types, pairs=pairs)
    df = paren_pheno.execute(lm, args.cid, spacing, stencil=stencil, num_offsets=args.num_offsets)
    df.to_csv(args.out_csv, index=False)
