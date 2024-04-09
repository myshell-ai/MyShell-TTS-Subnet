from datasets import load_dataset
import librosa
import soundfile as sf
import os

dataset = load_dataset("vctk", trust_remote_code=True)['train']
wav_folder_name = 'prepared_data'
#dataset.cleanup_cache_files()
#print(dataset)

def p225(audio_dict:dict):
    return audio_dict['speaker_id'] == 'p225'

p225_data = dataset.filter(p225, num_proc=8)
#print('---------------')
print(p225_data)


out_file = open('metadata.list', 'w')
for each_audio in p225_data:
    npAudioData, fSr = sf.read(each_audio['file'])
    print('------------')
    print(npAudioData, fSr)
    sWavFilename = each_audio['text_id'] + each_audio['speaker_id'] + each_audio['file'].split('/')[-1].split('.')[0].split('_')[-1] + '.wav'
    sWavFolderName = os.path.join(os.getcwd(), wav_folder_name)
    sWavFilename = os.path.join(sWavFolderName, sWavFilename)
    sf.write(sWavFilename, npAudioData, fSr)
    out_file.writelines(sWavFilename+'|EN-US|EN|'+each_audio['text']+'\n')
    
out_file.close()

