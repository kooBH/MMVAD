import glob
import numpy as np
import subprocess
import tqdm
import os
from collections import defaultdict, OrderedDict
import argparse

def calculate_DER(score_file, rttm_file_path, rttm_save_path, session='all'):
	out = subprocess.check_output(['perl', '../SD_VAD_Embed_SC/tools/SCTK-2.4.12/src/md-eval/md-eval.pl', '-c 0.25', '-s %s'%(rttm_save_path), '-r ' + rttm_file_path])
	out = out.decode('utf-8')
	DER, MS, FA, SC = map(float, out.split('/')[:4])
	print(session, ": DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%"%(DER, MS, FA, SC))
	score_file.write(session+ " : DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"%(DER, MS, FA, SC))
	score_file.flush()


def res_dict_writer(outs, filename, speaker_id, start, res_dict):
	B, _, T = outs.shape
	for b in range(B):
		for t in range(T):
			n = max(speaker_id[b,:])
			for i in range(n):
				id = speaker_id[b,i]
				name = filename[b]
				out = outs[b,i,t]
				t0 = start[b]
				res_dict[str(name) + '-' + str(id)][t0 + t].append(out)
	return res_dict

def prob2bin(rttm_save_path, res_dict, duration_unit=0.04):
	rttm = open(rttm_save_path, "w")
	for filename in tqdm.tqdm(res_dict):
		name, speaker_id =filename.split('-')
		probs = res_dict[filename]
		ave_probs = []
		for key in probs:
			ave_probs.append(np.mean(probs[key]))
		probs = ave_probs
		start, duration = 0, 0
		for i, prob in enumerate(probs):
			if prob == 1:
				duration += duration_unit
			else:
				if duration != 0:
					line =f"SPEAKER {name} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
					rttm.write(line)
					duration = 0
				start = i * duration_unit
		if duration != 0:
			line = f"SPEAKER {name} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
			rttm.write(line)
	rttm.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--input_dir',type=str,required=True)
	parser.add_argument('-o','--output_dir',type=str,required=True)
	args = parser.parse_args()

	#output_dir = "../data/eval_AMI_test_1/post_VVAD"
	output_dir = args.output_dir

	os.makedirs(output_dir, exist_ok=True)
	#! Set here to VVAD results .npy file
	#file_list = glob.glob('../data/eval_AMI_test_1/darization/vad/*.npy')
	file_list = glob.glob(args.input_dir + "/*.npy")
	if len(file_list) != 96:
		print(f"files : {len(file_list)}")
		raise("The number of files is not 96!")
	res_dict = defaultdict(lambda: defaultdict(list))
	for file in file_list:
		label_vad = np.load(file)
		res_dict = res_dict_writer(label_vad[None], (file.split('/')[-1][:-4],), np.arange(1,5)[None], np.array([0.0,]), res_dict)

	rttm_save_path = os.path.join(output_dir,'all.rttm')
	rttm_file_path = '/home/nas/user/Uihyeop/DB/AMI_IITP/rttms_ow/all.rttm'
	score_file = open(os.path.join(output_dir,'score_final.txt'), "w")

	label_vad_bin = prob2bin(rttm_save_path, res_dict, 0.6)
	calculate_DER(score_file, rttm_file_path, rttm_save_path)
	
