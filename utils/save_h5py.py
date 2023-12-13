import h5py
import mat73



def main():

    test_dl = mat73.loadmat('/data/data42/LiuCongcong/T1rho/data/Brain_glioma/glioma2/raw_org2.mat')['raw']

    h_kspce = h5py.File('/data/data42/LiuCongcong/T1rho/data/Brain_glioma/glioma2/raw_org2.h5', 'w')
    dset_kspace = h_kspce.create_dataset('raw', data=test_dl)
    h_kspce.close()

    # csm = mat73.loadmat('/data/data42/LiuCongcong/T1rho/data/new_data/maps.mat')['maps']

    # maps = h5py.File('/data/data42/LiuCongcong/T1rho/data/new_data/maps.h5', 'w')
    # dset_kspace = maps.create_dataset('maps', data=csm)
    # maps.close()


if __name__ == "__main__":
    main()