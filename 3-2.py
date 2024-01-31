import SimpleITK as sitk
import ants
from scipy import stats
import numpy as np
import imageio
from PIL import Image
def read_image(path):
    return ants.image_read(path)

templates = [read_image(f"data/ct_train_100{i}_imageROI.nii") for i in range(1,10)]
target =read_image("data/ct_train_1010_imageROI.nii")
Heart_labels = [read_image(f"data/ct_train_100{i}_Heart.nii") for i in range(1,10)]
Ao_labels = [read_image(f"data/ct_train_100{i}_Ao.nii") for i in range(1,10)]
LA_labels = [read_image(f"data/ct_train_100{i}_LA.nii") for i in range(1,10)]
LV_labels = [read_image(f"data/ct_train_100{i}_LV.nii") for i in range(1,10)]
LV_Myo_labels = [read_image(f"data/ct_train_100{i}_LV_Myo.nii") for i in range(1,10)]
RA_labels = [read_image(f"data/ct_train_100{i}_RA.nii") for i in range(1,10)]
RV_labels = [read_image(f"data/ct_train_100{i}_RV.nii") for i in range(1,10)]
PA_labels = [read_image(f"data/ct_train_100{i}_PA.nii") for i in range(1,10)]

labels = [Heart_labels,Ao_labels,LA_labels,LV_labels,LV_Myo_labels,RA_labels,RV_labels,PA_labels]
att = ['data/ct_train_1010_Heart.nii','data/ct_train_1010_Ao.nii','data/ct_train_1010_LA.nii','data/ct_train_1010_LV.nii','data/ct_train_1010_LV_Myo.nii','data/ct_train_1010_RA.nii','data/ct_train_1010_RV.nii','data/ct_train_1010_PA.nii']
sli = [50,100,60,50,40,40,30,130]
for i in range(0,1):
    result = []
    for template,label in zip(templates,labels[i]):
        trans=ants.registration(fixed=target,moving=template,type_of_transform="SyN")
        imageio.imsave('LA_{}.jpg'.format(len(result)), Image.fromarray(np.where(ants.apply_transforms(fixed=target,moving=label,transformlist=trans['fwdtransforms'])[:,:,sli[i]]>0,255,0).astype('uint8')))
        result.append(ants.apply_transforms(fixed=target,moving=label,transformlist=trans['fwdtransforms']))
    # vote = stats.mode(np.stack([r.numpy() for r in result]),axis=0,keepdims=True)[0]
    # final = ants.from_numpy(vote[0], origin = target.origin, spacing=target.spacing, direction=target.direction)
    # imageio.imsave('origin_{}.jpg'.format(i), Image.fromarray(np.where(read_image(att[i])[:,:,sli[i]]>0,255,0).astype('uint8')))
    # imageio.imsave('part_{}.jpg'.format(i), Image.fromarray(np.where(final[:,:,sli[i]]>0,255,0).astype('uint8')))