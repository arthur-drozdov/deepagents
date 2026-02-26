import sys
import json
import librosa

def load_wav_floats(wav_path):
    try:
        audio, _ = librosa.load(wav_path, sr=24000)
        # Output as a compact JSON list directly to stdout
        json.dump(audio.tolist(), sys.stdout, separators=(',', ':'))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: get_ref_floats.py <path_to_wav>", file=sys.stderr)
        sys.exit(1)

    load_wav_floats(sys.argv[1])
