import ollama
import whisper
import warnings

# Ignore FP16 warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


def process_audio(audio_path):
    """Transcribes audio and generates a summary using Whisper and LLaMA 3.2:1B.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        dict: Contains transcription, language, and summary.
    """
    # === Whisper: Audio -> Text ===
    model = whisper.load_model('base')  # Load Whisper model
    result = model.transcribe(audio_path)  # Transcribe audio
    transcribed_text = result["text"]
    transcribed_language = result["language"]

    # === LLaMA: Text -> Summarized Text ===
    summary_prompt = (
        f"Summarize the following transcription into concise key points:\n\n"
        f"{transcribed_text}"
    )
    
    # Use the llama3.2:1b model locally
    summary_response = ollama.chat(model="llama3.2:1b", messages=[
        {"role": "user", "content": summary_prompt}
    ])

    # Extract the summary
    transcribed_summary = summary_response['message']['content']

    return {
        'transcription': transcribed_text,
        'language': transcribed_language,
        'summary': transcribed_summary
    }
