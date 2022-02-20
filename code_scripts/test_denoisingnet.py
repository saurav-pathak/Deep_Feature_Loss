################## Change modules here
from models_mod_enet_new import *
from load_data_lib_enet_new import *
#from models_mod import *
#from load_data_lib import *

import sys, getopt
cuda = torch.device('cuda:1')

is_val = False # Check data train or validation

if is_val:
    dtype = 'val'
else:
    dtype = 'train'

#folder_last_name = 'denoised_modi_lib_enet_withoutSpecPipe_stsa_silu_latest'
#folder_last_name = 'denoised_v1_lib_latest'
#folder_last_name = 'denoised_modi_lib_latest'
#folder_last_name = 'denoised_modi_lib_enet_withoutSpecPipe_stsa_sisdr_silu_R14S12'
#folder_last_name = 'denoised_modi_lib_R14'
folder_last_name = 'denoised_modi_lib_enet_withoutSpecPipe_stsa_sisdr_silu_latest'
root_dir_metricGAN = '/home/sauravpathak/MetricGAN/dataset'
root_dir_NSDTSEA = '/home/sauravpathak/data/NSDTSEA'

is_metricGAN = True # check timit or voice bank
is_demand = True # check noise profile

if is_demand:
    suffix = '_demand'
else:
    suffix = ''

if is_metricGAN:
    root_dir = root_dir_metricGAN
else:
    root_dir = root_dir_NSDTSEA

validation_folder = '{}/{}set_noisy{}'.format(root_dir, dtype, suffix)
#denoising_model_path = "./models/denoising_model_modi_lib_enet_withoutSpecPipe_stsa_silu_latest.pth" # change model parameters
#denoising_model_path = "./models/denoising_model_v1_lib_latest.pth" # change model parameters
#denoising_model_path = "./models/denoising_model_modi_lib_latest.pth" # change model parameters
#denoising_model_path = "./models/denoising_model_modi_lib_enet_withoutSpecPipe_stsa_sisdr_silu_R14S12.pth" # change model parameters
#denoising_model_path = './models/denoising_model_modi_lib_R14.pth' # change model parameters
denoising_model_path = "./models/denoising_model_modi_lib_enet_withoutSpecPipe_stsa_sisdr_silu_latest.pth" # change model parameters

print("\n", flush=True)
print('Input Data folder is : "' + validation_folder + '/"', flush=True)
print('Denoising model path is : "' + denoising_model_path + '/"', flush=True)
print("Denoised outputs will be in the folder : " + validation_folder + "_denoised", flush=True)

if validation_folder[-1] == '/':
	validation_folder = validation_folder[:-1]

if not os.path.exists(validation_folder + '_' + folder_last_name):
	os.mkdir(validation_folder + '_' + folder_last_name)

frequency = 16000

# Loading of Data
validation_set = loadNoisyDataList(validationFolder = validation_folder)
validation_set = loadNoisyData(validation_set)


# Loading the saved model
include_spec_pipe = False # check whether to include spectrogram processing pipeline
is_silu = True # check activation
denoising_net_activation = 'silu' if is_silu else 'lrelu'
denoising_model = DenoisingNet(activation=denoising_net_activation, include_spec_pipe=include_spec_pipe).cuda(cuda)
#denoising_model = DenoisingNet().cuda(cuda)
denoising_model.load_state_dict(torch.load(denoising_model_path))
#denoising_model.eval()

######################################################## Spectrogram Extraction #########################################################

def get_spectrogram_db(raw_audio, sr = 16000, n_fft=2048, hop_length=512):
	raw_audio = raw_audio.cpu().detach().numpy()
	wav = np.squeeze(raw_audio)
	S = librosa.stft(wav, n_fft=n_fft)
	S = np.abs(S)
	S = librosa.amplitude_to_db(S, ref=np.max)
	return S

def spec_to_image(spec, eps=1e-6):
	mean = spec.mean()
	std = spec.std()
	spec_norm = (spec - mean) / (std + eps)
	spec_min, spec_max = spec_norm.min(), spec_norm.max()
	spec_scaled = 1 * (spec_norm - spec_min) / (spec_max - spec_min)
	return spec_scaled

def input_to_eff_net(raw_audio):
	S = spec_to_image(get_spectrogram_db(raw_audio))[np.newaxis,...]
	S = torch.as_tensor(S, device = cuda)
	S = torch.unsqueeze(S, 0)
	return S

########### Running on Validation Set ####################
print("\n---------------- Evaluation of validation dataset started ----------------------------\n")
for i in tqdm(range(0, len(validation_set['innames']))):

	index = i
	input_data = validation_set['inaudio'][index]
	input_data = torch.tensor(input_data).cuda(cuda)
#	input_spec = input_to_eff_net(input_data)
	input_spec = None
	enhanced_data = denoising_model(input_data, input_spec)
	enhanced_data = enhanced_data.detach().cpu().numpy()
	enhanced_data = np.reshape(enhanced_data, -1)
	wavfile.write("%s_%s/%s" % (validation_folder, folder_last_name, validation_set['shortnames'][i]), frequency, enhanced_data)

print("\n---------------- Evaluation of validation dataset ended ----------------------------\n")

