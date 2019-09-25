import numpy as np
from scipy.stats import mode, kurtosis, skew
from argparse import ArgumentParser
import warnings

from cip_python.phenotypes import Phenotypes
from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser


class ParenchymaPhenotypesProgression(Phenotypes):
    """General purpose class for generating parenchyma-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:

        TO BE FILLED

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

    def __init__(self, chest_regions=None, chest_types=None, pairs=None, pheno_names=None):
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

        self.deprecated_phenos_ = ['NormalParenchyma', 'PanlobularEmphysema',
                                   'ParaseptalEmphysema', 'MildCentrilobularEmphysema',
                                   'ModerateCentrilobularEmphysema', 'SevereCentrilobularEmphysema',
                                   'MildParaseptalEmphysema']

        Phenotypes.__init__(self)

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """

        # Get phenotypes list from ChestConventions
        c = ChestConventions()
        names = c.ParenchymaPhenotypeNames

        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def transition_matrix(self,x,y,t):
        """
            Calculates the transition matrix from the two sets of paired samples for a certain trehsold (t)

            0: Below the threshold
            1: Beyond the thhreshold

            Format output: outXY:
                X: Time 1
                Y: Time 2
        """

        assert len(x) == len(y), "Samples are not paired"
        n = len(x)
        out00 = (x<=t)*(y<=t)
        out01 = (x<=t)*(y>t)
        out10 = (x>t)*(y<=t)
        out11 = (x>t)*(y>t)

        return out00, out11, out10, out11

    def execute(self, lm, cid, spacing, ct1=None,  ct2=None, chest_regions=None, chest_types=None, pairs=None, pheno_names=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        The following values are computed for the specified structures.
            TO BE FILLED

        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume

        ct : array, shape ( X, Y, Z )
            The 3D CT image array

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
        if ct1 is not None:
            assert len(ct1.shape) == len(lm1.shape), "CT1 and label map are not the same dimension"
        if ct2 is not None:
            assert len(ct2.shape) == len(lm2.shape), "CT2 and label map are not the same dimension"

        dim1 = len(lm1.shape)
        if ct1 is not None:
            for i in range(0, dim1):
                assert ct1.shape[0] == lm1.shape[0], "Disagreement in CT1 and label map dimension"
        dim2 = len(lm2.shape)
        if ct2 is not None:
            for i in range(0, dim2):
                assert ct2.shape[0] == lm2.shape[0], "Disagreement in CT2 and label map dimension"

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

        # Deal with wildcard: defaul to main pheno list if wildcard is provided
        if c.GetChestWildCardName() in phenos_to_compute:
            phenos_to_compute = self.pheno_names_

        # Check validity of phenotypes
        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
                "Invalid phenotype name " + pheno_name

        # Warn for phenotpyes that have been deprecated
        for p in phenos_to_compute:
            if p in self.deprecated_phenos_:
                warnings.warn('{} is deprecated. Use TypeFrac instead.'. \
                              format(p), DeprecationWarning)

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
                self.add_pheno_group(ct1, ct2, mask_region, mask_region,
                                     mask_region, mask_region,
                                     None, c.GetChestRegionName(r),
                                     c.GetChestWildCardName(), phenos_to_compute)
        for t in ts:
            if t != 0:
                mask_type = parser.get_mask(chest_type=t)
                self.add_pheno_group(ct1, ct2, mask_type, mask_type, None, mask_type,
                                     c.GetChestWildCardName(),
                                     c.GetChestTypeName(t), phenos_to_compute)
        if ps.size > 0:
            for i in range(0, ps.shape[0]):
                if not (ps[i, 0] == 0 and ps[i, 1] == 0):
                    mask = parser.get_mask(chest_region=int(ps[i, 0]),
                                           chest_type=int(ps[i, 1]))
                    if 'TypeFrac' in phenos_to_compute:
                        mask_region = \
                            parser.get_mask(chest_region=int(ps[i, 0]))
                    else:
                        mask_region = None

                    self.add_pheno_group(ct1, ct2, mask1, mask2, mask_region,
                                         None, c.GetChestRegionName(int(ps[i, 0])),
                                         c.GetChestTypeName(int(ps[i, 1])), phenos_to_compute)

        return self._df

    def add_pheno_group(self, ct1, ct2, mask1, mask2, mask_region, mask_type, chest_region, chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method.

        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
            The 3D CT image array

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
        print ("Computing phenos for region %s and type %s" % (chest_region, chest_type))
        if ct is not None:
            ct_mask1 = ct1[mask1]
            ct_mask2 = ct2[mask2]

            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                    "Invalid phenotype name " + pheno_name
                pheno_val = None
                if pheno_name == 'TMLAA950' and mask_sum > 0:
                    self.transition_matrix()
                elif pheno_name == 'TMLAA925' and mask_sum > 0:
                    self.transition_matrix()

                if pheno_val is not None:
                    self.add_pheno([chest_region, chest_type],
                                   pheno_name, pheno_val)

        if 'TypeFrac' in phenos_to_compute and mask is not None and \
                        mask_region is not None:
            denom = np.sum(mask_region)
            if denom > 0:
                pheno_val = float(np.sum(mask)) / float(denom)
                self.add_pheno([chest_region, chest_type],
                               'TypeFrac', pheno_val)


if __name__ == "__main__":
    desc = """Generates progression parenchyma phenotypes given input CT and segmentation \
    data. Note that, in general, specified chest-regions, chest-types, and \
    region-type pairs indicate the parts of the image over which the requested \
    phenotypes are computed. An exception is the TypeFrac phenotype; in that 
    case, a region-type pair must be specified. The TypeFrac phenotype is \
    computed by calculating the fraction of the pair's chest-region that is \
    covered by the pair's chest-type. For example, if a user is interested in \
    the fraction of paraseptal emphysema in the left lung, he/she would \
    specify the TypeFrac phenotype and the LeftLung,ParaseptalEmphysema \
    region-type pair."""

    parser = ArgumentParser(description=desc)

    parser.add_argument('--in_ct1', '-in_ct1', required=True, help='Input CT file in time 1')
    parser.add_argument('--in_ct2', '-in_ct2', required=True, help='Input CT file in time 2')

    parser.add_argument('--in_lm1', '-in_lm1', required=True, help='Input label map containing structures of interest in time 1')
    parser.add_argument('--in_lm2', '-in_lm2', required=True, help='Input label map containing structures of interest in time 2')

    parser.add_argument('--out_csv', '-out_csv', required=True,
                        help='Output csv file in which to store the computed dataframe')

    parser.add_argument('--cid', '-cid', required=True, help='Case id')

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

    lm1, lm_header1 = image_io.read_in_numpy(args.in_lm1)
    lm2, lm_header2 = image_io.read_in_numpy(args.in_lm2)

    ct1, ct_header1 = image_io.read_in_numpy(args.in_ct1)
    ct2, ct_header2 = image_io.read_in_numpy(args.in_ct2)

    spacing = lm1_header['spacing']

    pairs = None
    if args.pairs is not None:
        tmp = args.pairs.split(',')
        assert len(tmp) % 2 == 0, 'Specified pairs not understood'
        pairs = []
        for i in range(0, len(tmp) / 2):
            pairs.append([tmp[2 * i], tmp[2 * i + 1]])

    paren_pheno = ParenchymaPhenotypesProgression(chest_regions=regions, chest_types=types, pairs=pairs)

    ########################################
    # Debugging
    if options.in_file_path is None:
        print("################")
        print("DEBUGGING MODE!")
        print("################")

        scans_dir = '/Users/gvegas/data/temp'
        cid1 = '19020F_INSP_STD_UIA_COPD'
        cid2 = '19020F_INSP_STD_UIA_COPD'
        ct1 = scans_dir + '/' + cid1 + '.nrrd'
        lm1 = scans_dir + '/' + cid1 + '_partialLungLabelMap.nrrd'
        ct1 = scans_dir + '/' + cid2 + '.nrrd'
        lm2 = scans_dir + '/' + cid2 + '_partialLungLabelMap.nrrd'
        regions = "WholeLung,LeftLung,RightLung,UpperThird,MiddleThird,LowerThird,LeftUpperThird,LeftMiddleThird,LeftLowerThird,RightUpperThird,RightMiddleThird,RightLowerThird"
        regions = regions.split(',')
        types = None
        pairs = None

        chest_regions = regions

        paren_pheno = ParenchymaPhenotypesProgression(chest_regions=regions, chest_types=types, pairs=pairs)
        self = paren_pheno

        lm, cid, spacing, ct = None, chest_regions = None, chest_types = None, pairs = None, pheno_names = None

    df = paren_pheno.execute(lm1, lm2, args.cid, spacing, ct1=ct1, ct2=ct2)


    df = paren_pheno.execute(lm, args.cid, spacing, ct=ct)

    df.to_csv(args.out_csv, index=False)
