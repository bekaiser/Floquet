




import runpy
import os
import shutil



u_path = '/home/bryan/git_repos/Floquet/base_flow/u/'
uz_path = '/home/bryan/git_repos/Floquet/base_flow/uz/'
uzz_path = '/home/bryan/git_repos/Floquet/base_flow/uzz/'
v_path = '/home/bryan/git_repos/Floquet/base_flow/v/'
vz_path = '/home/bryan/git_repos/Floquet/base_flow/vz/'
vzz_path = '/home/bryan/git_repos/Floquet/base_flow/vzz/'
b_path = '/home/bryan/git_repos/Floquet/base_flow/b/'
bz_path = '/home/bryan/git_repos/Floquet/base_flow/bz/'
bzz_path = '/home/bryan/git_repos/Floquet/base_flow/bzz/'

paths = {'u_path':u_path, 'uz_path':uz_path, 'uzz_path':uzz_path, 
         'v_path':v_path, 'vz_path':vz_path, 'vzz_path':vzz_path, 
         'b_path':b_path, 'bz_path':bz_path, 'bzz_path':bzz_path}

folder = (paths['u_path'],paths['uz_path'],paths['uzz_path'],
            paths['v_path'],paths['vz_path'],paths['vzz_path'],
            paths['b_path'],paths['bz_path'],paths['bzz_path'],)

for j in range(0,9):
    for the_file in os.listdir(folder[j]):
        file_path = os.path.join(folder[j], the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
