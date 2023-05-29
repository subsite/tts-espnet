
import os, sys, soundfile, time
from espnet2.bin.tts_inference import Text2Speech

model = "kan-bayashi/ljspeech_vits"

if len(sys.argv) != 3:
    exit("USAGE: tts.py infile output_dir")

infile = sys.argv[1]
outdir = sys.argv[2]

try:
    with open(infile) as f:
        text = f.read()
        infile_name = os.path.basename(f.name)
except:
    exit(f"Failed to open file {infile}")

outfile = os.path.join(outdir, infile_name.split('.')[0] + '.wav')

print(f"Saving speech to {outfile}")



start = time.time()
try:
    text2speech = Text2Speech.from_pretrained(model_tag=model, device="cuda")
    wav = text2speech(text)["wav"]
except UserWarning as w:
    print("warn")


soundfile.write(outfile, wav.view(-1).cpu().numpy(), text2speech.fs, "PCM_16")

rtf = (time.time() - start) / (len(wav) / text2speech.fs) # Real Time Factor

print(f"Done in {(time.time() - start):.2f} sec, RTF: {rtf:.2f}")