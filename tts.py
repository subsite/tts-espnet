import os, sys, math, soundfile, time, torch
from espnet2.bin.tts_inference import Text2Speech

model = "kan-bayashi/ljspeech_vits"
chunk_max_chars = 700

if len(sys.argv) != 3:
    exit("USAGE: tts.py infile [ output_dir_only | dir_and_file.wav ]")

infile = sys.argv[1]
outfile = sys.argv[2]

try:
    with open(infile) as f:
        text_lines = f.readlines()
        infile_name = os.path.basename(f.name)
except:
    exit(f"Failed to open file {infile}")

# Split text into chunks
text_chunks = ['']
for line in text_lines:
    #print(f"    line: {line[:20].strip()}...")
    if len(line) > chunk_max_chars:
        exit(f"FATAL: line length {len(line)} exceeds chunk_max_chars")
    if len(text_chunks[-1] + line.strip()) > chunk_max_chars:
        text_chunks.append('')
    text_chunks[-1] += line


# output file is named like input file if only dir is set
if outfile[-4:] != ".wav":
    outfile = os.path.join(outfile, infile_name.split('.')[0] + '.wav')

print(f"Saving {len(text_chunks)} text chunks to speech as {outfile}")

start = time.time()

text2speech = Text2Speech.from_pretrained(
    model_tag=model, 
    device="cuda")

waw_chunks = []
cur_chunk = 1
with torch.no_grad():
    for chunk in text_chunks:
        print(f"\rChunk {cur_chunk}/{len(text_chunks)} {round(cur_chunk/len(text_chunks)*100)}%", end="", flush=True)
        cur_chunk += 1    
        waw_chunks.append(text2speech(chunk)["wav"])
    
print("Joining chunks to wav...")
wav = torch.cat(waw_chunks)
soundfile.write(outfile, wav.view(-1).cpu().numpy(), text2speech.fs, "PCM_16")

rtf = (time.time() - start) / (len(wav) / text2speech.fs) # Real Time Factor

print(f"\nDone in {(time.time() - start):.2f} sec, RTF: {rtf:.2f}")
