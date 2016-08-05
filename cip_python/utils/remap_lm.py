import numpy as np
from optparse import OptionParser
from ..common import ChestConventions
from ..utils import RegionTypeParser
from ..input_output import ImageReaderWriter

def remap_lm(lm, region_maps=None, type_maps=None, pair_maps=None):
    """Overwrites values in an input label map using the specified mappings.

    Parameters
    ----------
    lm : array
        The input data. Each value is assumed to be an unsigned short (16 bit)
        data type, where the least significant 8 bits encode the chest region,
        and the most significant 8 bits encode the chest type.

    region_maps : list of lists, optional
        Each element of the list is a 2-element list or a list with two
        elements, where the individual elements are strings indicating chest
        regions. The 2D lists / tuples indicate the mappings to be performed.
        E.g. [('LeftLung', 'WholeLung')] indicates that all occurrences of
        'LeftLung' should be replaced with 'WholeLung'. All type information
        is preserved. 

    type_maps : list of lists, optional
        Each element of the list is a 2-element list or a list with two
        elements, where the individual elements are strings indicating chest
        types. The 2D lists / tuples indicate the mappings to be performed. E.g.
        [('Airway', 'UndefinedType')] indicates that all occurrences of
        'Airway' should be replaced with 'UndefinedType'. All region information
        is preserved. 

    pairs_maps : list of lists of lists, optional
        Each element of the list is a 2-element list: the firs element is itself
        a 2-element list indicating a region-type pair; the second element is a
        2-element region-type pair to be mapped to. String names are used to
        designate the regions and types. E.g. [['LeftLung', 'Vessel'],
        ['WholeLung', 'UndefinedType']] will map all occurrences of the
        'LeftLung'-'Vessel' pair to 'WholeLung'-'UndefinedType'.

    Returns
    -------
    remapped_lm : array
        A label map with the same dimensions as the input with labelings
        redefined according to the specified rules.
    """
    remapped_lm  = np.copy(lm)
    parser = RegionTypeParser(lm)

    lm_types = lm >> 8
    lm_regions = lm - (lm >> 8 << 8)

    c = ChestConventions()
    if region_maps is not None:
        for m in xrange(0, len(region_maps)):
            assert len(region_maps[m]) == 2, "Mapping not understood"
            mask = \
                parser.get_mask(chest_region=\
                       c.GetChestRegionValueFromName(region_maps[m][0]))
            lm_regions[mask] = \
                c.GetChestRegionValueFromName(region_maps[m][1])

    if type_maps is not None:
        for m in xrange(0, len(type_maps)):
            assert len(type_maps[m]) == 2, "Mapping not understood"
            mask = \
                parser.get_mask(chest_type=\
                       c.GetChestTypeValueFromName(type_maps[m][0]))
            lm_types[mask] = \
                c.GetChestTypeValueFromName(type_maps[m][1])

    if pair_maps is not None:
        for m in xrange(0, len(pair_maps)):
            assert len(pair_maps[m]) == 2, "Mapping not understood"
            assert len(pair_maps[m][0]) == 2, "Mapping not understood"
            assert len(pair_maps[m][1]) == 2, "Mapping not understood"            
            mask = \
                parser.get_mask(chest_region=\
                       c.GetChestRegionValueFromName(pair_maps[m][0][0]),
                    chest_type=\
                       c.GetChestTypeValueFromName(pair_maps[m][0][1]))
            lm_regions[mask] = \
                c.GetChestRegionValueFromName(pair_maps[m][1][0])            
            lm_types[mask] = \
                c.GetChestTypeValueFromName(pair_maps[m][1][1])

    remapped_lm = (lm_types << 8) | lm_regions

    return remapped_lm

if __name__ == "__main__":
    desc = """Remap chest regions, chest types, or region-type pairs in an \
    input labelmap using specified mapping rules"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_lm',
                      help='Input label map',
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--out_lm',
                      help='Output label map',
                      dest='out_lm', metavar='<string>', default=None)
    parser.add_option('--region_maps',
                      help='Chest region mappings. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired mappings. The first string value is interpreted \
                      as the first from-region, the second string value is \
                      interpreted as the first to-region, the third string \
                      value is interpreted as the second from-region, etc. \
                      E.g.: LeftLung,WholeLung,RightLung,WholeLung will map \
                      all voxels having a chest region value of LeftLung or \
                      RightLung to a chest region value of WholeLung. Whatever \
                      type value is contained in those voxels will be \
                      preserved.',
                      dest='region_maps', metavar='<string>', default=None)
    parser.add_option('--type_maps',
                      help='Chest type mappings. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired mappings. The first string value is interpreted \
                      as the first from-type, the second string value is \
                      interpreted as the first to-type, the third string \
                      value is interpreted as the second from-type, etc. \
                      E.g.: Vessel,NormalParenchyma will map all voxels having \
                      a chest type value of Vessel to chest type value of \
                      NormalParenchym. Whatever type region is contained in \
                      those voxels will be preserved.',
                      dest='type_maps', metavar='<string>', default=None)    
    parser.add_option('--pair_maps',
                      help='Chest region-type pair mappings. Should be \
                      specified as a common-separated list of string values \
                      indicating the desired mappings. The first two string \
                      values are interpreted as the first from-pair, the \
                      second pair of string values are interpreted as the \
                      first to-pair, the third pair of string values are \
                      interpreted as the second from-pair, etc. E.g.: \
                      LeftLung,Vessel,WholeLung,UndefinedType will map all \
                      voxels having chest region-type pair values of \
                      LeftLung, Vessel to chest region-type values \
                      corresponding to WholeLung, UndefinedType.',
                      dest='pair_maps', metavar='<string>', default=None)    

    (options, args) = parser.parse_args()

    region_maps = None
    type_maps = None
    pair_maps = None

    if options.region_maps is not None:
        region_maps = []
        split = options.region_maps.split(',')
        assert len(split)%2 == 0, \
            "Improperly specified region mappings"
        inc = 0
        while inc < len(split):
            tmp = []
            tmp.append(split[inc])
            inc = inc+1
            tmp.append(split[inc])
            inc = inc+1
            region_maps.append(tmp)

    if options.type_maps is not None:
        type_maps = []
        split = options.type_maps.split(',')
        assert len(split)%2 == 0, \
            "Improperly specified type mappings"
        inc = 0
        while inc < len(split):
            tmp = []
            tmp.append(split[inc])
            inc = inc+1
            tmp.append(split[inc])
            inc = inc+1
            type_maps.append(tmp)

    if options.pair_maps is not None:
        pair_maps = []
        split = options.pair_maps.split(',')
        assert len(split)%4 == 0, \
            "Improperly specified pair mappings"
        inc = 0
        while inc < len(split):
            tmp_from = []
            tmp_from.append(split[inc])
            inc = inc+1
            tmp_from.append(split[inc])
            inc = inc+1
            tmp_to = []
            tmp_to.append(split[inc])
            inc = inc+1
            tmp_to.append(split[inc])
            inc = inc+1
            pair = [tmp_from, tmp_to]            
            pair_maps.append(pair)

    lm_io = ImageReaderWriter()
    lm_in, header = lm_io.read_in_numpy(options.in_lm)
    #lm_in, header = nrrd.read(options.in_lm)

    remapped_lm = remap_lm(lm_in, region_maps=region_maps,
                           type_maps=type_maps, pair_maps=pair_maps)

    if options.out_lm is not None:
      lm_io.write_from_numpy(remapped_lm,header,options.out_lm)
  
#    if options.out_lm is not None:
#        del header['data file']
#        nrrd.write(options.out_lm, remapped_lm, options=header)

