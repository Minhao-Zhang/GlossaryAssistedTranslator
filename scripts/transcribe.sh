docker run --gpus all -it -v "$(pwd):/app" ghcr.io/jim60105/whisperx:large-v3-en \
  -- --output_format srt --output_dir . --language en \
  --hf_token YOUR_HUGGINGDACE_TOKEN \
  --diarize --min_speakers 2 --max_speakers 2 YOUR_AUDIO_FILE