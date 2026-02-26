import numpy as np
import os
import matplotlib.pyplot as plt

cases = ["experiments_bash/results/remove_all",
         "experiments_bash/results/remove_filter",
        #  "experiments_bash/results/sqrt2_per_3",
         "experiments_bash/results/sqrt2_per_2"]

scenes = ["replica_room0",
         "replica_room1",
         "replica_office0",
         "tum_freiburg1_desk",
         "tum_freiburg2_xyz",
         "tum_freiburg3_long_office_household"]

iters = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for scene in scenes:
    for case in cases:
        avg_psnrs_iters = []
        for iter in iters:
            result_txt_path = os.path.join(case, scene, (str)(iter), "psnr_gaussian_splatting.txt")
            result_txt = open(result_txt_path, "r")
            lines = result_txt.readlines()
            
            psnrs = []
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                psnr = (float)(line.split()[1])
                psnrs.append(psnr)
                
            psnrs = np.array(psnrs)
            
            avg_psnr = psnrs.mean()
            avg_psnrs_iters.append(avg_psnr)
            
        plt.plot(iters, avg_psnrs_iters, label=case.split("/")[-1])
        print(f"{scene}/{iter}/ nofilter, per3, per2: {avg_psnrs_iters[0]},{avg_psnrs_iters[1]},{avg_psnrs_iters[2]}")
    plt.legend()
    plt.title(scene)
    plt.show()
    plt.cla()