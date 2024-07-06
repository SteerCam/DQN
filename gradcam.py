from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from stable_baselines3 import PPO,DQN
#from latest2SB3_ComAI_RL_imageComAIwithBestCollaborotorWithInterval import CustomEnv
import cv2
import numpy as np
image_path="detected.jpg"
model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (150, 150))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
## Note: input_tensor can be a batch tensor with several images!
#env = CustomEnv(traning=False)
#model = DQN(
#    "CnnPolicy",
#    env,
#    verbose=1,
#    train_freq=(8, "step"), #16 (4, "step")
#    gradient_steps= -1, #8
#    gamma=1,
#    exploration_fraction=0.5 , #0.375
#    exploration_final_eps=0.01,
#    target_update_interval=1000, #600
#    learning_starts=100000, #1000nvidia-smi
#    buffer_size=100000, #200000
#    batch_size=32,#,128,
#    learning_rate=1e-7,
#    tensorboard_log=None,
#    seed=2, 
#    device="cuda",
#    policy_kwargs=dict(net_arch=[150, 150]), #dict(net_arch=[256, 256])
#  )
#checkpoint_path="./best_model_4sofar/dqn_best_model_"+str(3000000)+"_steps"
#model.set_parameters(checkpoint_path)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [ClassifierOutputTarget(281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cv2.imwrite("./gradcam.png",visualization)
# You can also get the model outputs without having to re-inference
#model_outputs = cam.outputs