import sys,os
sys.path.insert(0, os.environ['ATS_BASE'] + '/repos/amanzi/tools/amanzi_xml/')
import amanzi_xml.utils.io
import amanzi_xml.utils.search as search
# import cPickle as pickle
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

for r,reg in enumerate(regions):
    base_forcing = '../../../../data/' + reg + '_spinup.h5'
    for j in ['02', '05']:#, '05', '1']:
        mesh = 'column_' + j + 'peat.exo'
        mesh_path = '../../../../data/' + mesh
        restart_path = '../../../../' + j + 'peat_freezeup/checkpoint_final.h5'
        direc = basepath + '/regions/' + reg + '/' + j + 'peat/spinup'
        if j == '0':
            xml = amanzi_xml.utils.io.fromFile(
            basepath + '/templates/template_spinup_min.xml')
        else:
            xml = amanzi_xml.utils.io.fromFile(
            basepath + '/templates/template_spinup.xml')
        print(direc)
        
        search.change_value(xml, ['cycle driver','end time'], 20)  
        search.change_value(xml, ['cycle driver','end time units'], 'noleap')  
        
        search.change_value(xml, ['checkpoint','times start period stop'], "{0, 1, -1}")
        search.change_value(xml, ['checkpoint','times start period stop units'], "noleap")
        
        search.change_value(xml, ['PKs','subsurface energy','boundary conditions','temperature','bottom','boundary temperature','function-constant', 'value'],264.15)
        
        search.change_value(xml, ['visualization','domain','times start period stop'], "{"+str(365*18+1)+", 1, -1}")
        search.change_value(xml, ['visualization','domain','times start period stop units'], "d")
        search.change_value(xml, ['visualization','snow','times start period stop'], "{"+str(365*18+1)+", 1, -1}") 
        search.change_value(xml, ['visualization','snow','times start period stop units'], "d")
        search.change_value(xml, ['visualization','surface','times start period stop'], "{"+str(365*18+1)+", 1, -1}") 
        search.change_value(xml, ['visualization','surface','times start period stop units'], "d")
        
        search.change_value(xml, ['regions','subsurface domain peat','region: labeled set','file'], mesh_path)
        search.change_value(xml, ['regions','subsurface domain mineral','region: labeled set','file'], mesh_path)
        search.change_value(xml, ['regions','surface face','region: labeled set','file'], mesh_path)
        search.change_value(xml, ['regions','bottom face','region: labeled set','file'], mesh_path)
        search.change_value(xml, ['mesh','domain','read mesh file parameters','file'], mesh_path)  
        search.change_value(xml, ['PKs','subsurface flow','initial condition','restart file'], restart_path)   
        search.change_value(xml, ['PKs','subsurface energy','initial condition','restart file'], restart_path)      
        search.change_value(xml, ['state','field evaluators','surface-incoming_shortwave_radiation','function','domain','function','function-tabular','file'], base_forcing)
        search.change_value(xml, ['state','field evaluators','surface-air_temperature','function','domain','function','function-tabular','file'], base_forcing)
        search.change_value(xml, ['state','field evaluators','surface-relative_humidity','function','domain','function','function-tabular','file'], base_forcing)
        search.change_value(xml, ['state','field evaluators','surface-wind_speed','function','domain','function','function-tabular','file'], base_forcing)
        search.change_value(xml, ['state','field evaluators','surface-precipitation_rain','function','domain','function','function-tabular','file'], base_forcing)
        search.change_value(xml, ['state','field evaluators','snow-precipitation','function','domain','function','function-tabular','file'], base_forcing)
            
        if not os.path.exists(direc):
            os.makedirs(direc)
        os.chdir(direc)
        cp_cmd = 'cp '+basepath+'/scripts/run.sh .'
        os.system(cp_cmd)
        amanzi_xml.utils.io.toFile(xml, 'permafrost_column.xml')
