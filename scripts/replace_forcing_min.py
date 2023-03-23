import sys, os
sys.path.insert(0, os.environ['ATS_BASE'] + '/repos/amanzi/tools/amanzi_xml/')
import amanzi_xml.utils.io
import amanzi_xml.utils.search as search
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--region', type=str,
                    help="region for modeling")

args = parser.parse_args()
basepath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
reg = str(args.region)
if args.region == None:
    regions= ['cd','cw','wd','ww']
else:
    regions = [reg]
subsoil_depths = ['-0.4', '-0.36', '-0.52', '-0.50']

for r,reg in enumerate(regions):
    base_forcing = '../../../../data/' + reg + '_control.h5'
    irrigation_forcing = '../../../../data/' + reg + '_irrigation.h5'
    subsoil = '{ 0.5, 0.5, ' + subsoil_depths[r] + '}'
    for j in ['0']:
        mesh = 'column_' + j + 'peat.exo'
        mesh_path = '../../../../data/' + mesh
        for mode in ['i', 'c']:
            direc = basepath + '/regions/' + reg + '/' + j + 'peat/' + mode
            xml = amanzi_xml.utils.io.fromFile(
                basepath  + '/templates/template_min.xml')
            print(direc)
            search.searchAndReplaceByName(xml,
                                          'regions/subsoil/region: plane/point=' + subsoil)
            search.searchAndReplaceByName(xml,
                                          'regions/subsurface domain peat/region: labeled set/file=' + mesh_path)
            search.searchAndReplaceByName(xml,
                                          'regions/subsurface domain mineral/region: labeled set/file=' + mesh_path)
            search.searchAndReplaceByName(xml,
                                          'regions/surface face/region: labeled set/file=' + mesh_path)
            search.searchAndReplaceByName(xml,
                                          'regions/bottom face/region: labeled set/file=' + mesh_path)
            search.searchAndReplaceByName(xml,
                                          'mesh/domain/read mesh file parameters/file=' + mesh_path)
            search.searchAndReplaceByName(xml,
                                          'state/field evaluators/surface-incoming_shortwave_radiation/function/domain/function/function-tabular/file=' + base_forcing)
            search.searchAndReplaceByName(xml,
                                           'state/field evaluators/surface-air_temperature/function/domain/function/function-tabular/file=' + base_forcing)
            search.searchAndReplaceByName(xml,
                                           'state/field evaluators/surface-relative_humidity/function/domain/function/function-tabular/file=' +  base_forcing)
            search.searchAndReplaceByName(xml,
                                           'state/field evaluators/surface-wind_speed/function/domain/function/function-tabular/file=' + base_forcing)
            if mode == 'i':
                search.searchAndReplaceByName(xml,
                                           'state/field evaluators/surface-precipitation_rain/function/domain/function/function-tabular/file=' + irrigation_forcing)
            else:
                search.searchAndReplaceByName(xml,
                                           'state/field evaluators/surface-precipitation_rain/function/domain/function/function-tabular/file=' + base_forcing)

            search.searchAndReplaceByName(xml,
                                           'state/field evaluators/snow-precipitation/function/domain/function/function-tabular/file=' + base_forcing)
            if not os.path.exists(direc):
                os.mkdir(direc)
            os.chdir(direc)
            cp_cmd = 'cp '+basepath+'/scripts/run.sh .'
            os.system(cp_cmd)
            amanzi_xml.utils.io.toFile(xml, 'permafrost_column.xml')
