import numpy as np
import pickle

def Matrix_rotate(alpha_deg, beta_deg, gamma_deg):
    al_rad = np.radians(alpha_deg)
    beta_rad = np.radians(beta_deg)
    gam_rad = np.radians(gamma_deg)
    X_rot = np.array([[1,0,0], [0,np.cos(al_rad),-np.sin(al_rad)], [0,np.sin(al_rad),np.cos(al_rad)]])
    Y_rot = np.array([[np.cos(beta_rad),0, np.sin(beta_rad)], [0,1,0], [-np.sin(beta_rad),0,np.cos(beta_rad)]])
    Z_rot = np.array([[np.cos(gam_rad),-np.sin(gam_rad),0], [np.sin(gam_rad),np.cos(gam_rad),0], [0,0,1]])
    Rotate = X_rot.dot(Y_rot.dot(Z_rot))
    return Rotate

def aug_rot(data, count_new_elem):
    #alphas = [-20 .. 20]
    #betas = [0 .. 720] (градусы)
    #gammas = [-20 .. 20]
    alpha  = -20
    beta = 0
    gamma = -20
    elements = np.zeros((count_new_elem, data.shape[0], data.shape[1], data.shape[2]))
    for i in range(count_new_elem):
        alpha += (40 / count_new_elem)
        beta += (720 / count_new_elem) # два оборота
        gamma += (40 / count_new_elem)
        R = Matrix_rotate(alpha, beta, gamma)
        for j in range(data.shape[0]):
            elements[i][j] = R.dot(data[j].T).T
    new_elements = elements.reshape((elements.shape[0] * elements.shape[1], elements.shape[2], elements.shape[3]))
    return np.concatenate((data, new_elements))

def merge_arrays(dict):
    merged_arr = np.zeros((1, 15, 3))
    for key, value in dict.items():
        print("key", key, "value", value)
        merged_arr = np.concatenate((merged_arr, np.array(value)))
    return merged_arr[1:]


def normalize(data, mode):
    if (mode == 'frame'):
        if (data.shape[1] == 3):
            var = (data[:, 0].std() + data[:, 1].std()) / 2
            means = [data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean()]
            data -= means
            data /= var
        elif (data.shape[1] == 2):
            var = (data[:, 0].std() + data[:, 1].std()) / 2
            means = [data[:, 0].mean(), data[:, 1].mean()]
            data -= means
            data /= var
    elif (mode == "sequence"):
        for frame in data:
            if (frame.shape[1] == 3):
                var = (frame[:, 0].std() + frame[:, 1].std()) / 2
                means = [frame[:, 0].mean(), frame[:, 1].mean(), frame[:, 2].mean()]
                frame -= means
                frame /= var
            elif (frame.shape[1] == 2):
                var = (frame[:, 0].std() + frame[:, 1].std()) / 2
                means = [frame[:, 0].mean(), frame[:, 1].mean()]
                frame -= means
                frame /= var
    return data

save_dir = 'data_sets/set'

for cur_dir in ['13','14','15','86']:
    cur_dict = 'dicts_from_blender/dict{}.p'.format(cur_dir)
    cur_file = pickle.load(open(blender_file, "rb"))
    ready = aug_rot(normalize(merge_arrays(cur_file), "sequence"), 720)
    np.save(save_dir{}.format(cur_dir), ready)