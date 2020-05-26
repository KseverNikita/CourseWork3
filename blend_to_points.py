import bpy #библиотека для работы с Blender
import numpy as np
import pickle #библиотека для считывания и записи файлов
import os

joints = {
    0: "Head",
    1: "Spine",
    2: "RightShoulder",
    3: "RightArm",
    4: "RightForeArm",
    5: "LeftShoulder",
    6: "LeftArm",
    7: "LeftForeArm",
    8: "RHipJoint",
    9: "RightUpLeg",
    10: "RightLeg",
    11: "LHipJoint",
    12: "LeftUpLeg",
    13: "LeftLeg",
    14: "Hips",
    }

bvh_dirs = [13,14,15,86]
for cur_dir in bvh_dirs:
    file_list = []
    change_dir = 'change_bvhs/{}/'.format(cur_dir)
    for _, _, files in os.walk(change_dir):
        bvh_dict = {}
        for file in files:
            bpy.ops.import_anim.bvh(filepath=(change_dir + file))
            bpy.ops.object.mode_set(mode='POSE')
            max_frames = np.array(bpy.context.object.animation_data.action.fcurves)[0].keyframe_points.shape[0]
            bvh_dict[file] = []
            for i in range(0, max_frames):
                bpy.context.scene.frame_set(i)
                cur_frame = []
                for j in range(15):
                    cur_bone = bpy.context.object.pose.bones[joints[j]]
                    t = cur_bone.tail
                    cur_x, cur_y, cur_z = t[0], t[2], -t[1]
                    set_coords = [x,y,z]
                    cur_frame.append(set_coords)
                bvh_dict[file].append(cur_frame)
            bpy.ops.object.mode_set(mode='POSE') #mode="OBJECT"
            bpy.ops.object.delete()
        pickle.dump(bvh_dict, open('dicts_from_blender/dict{}.p'.format(cur_dir), "wb" ))