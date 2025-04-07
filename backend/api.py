from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
from whisper_llama import process_audio
from dotenv import load_dotenv

load_dotenv()  #this will load the environment variables from the .env file

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:5501", "http://localhost:5501"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "max_age": 3600
    }
})

@app.route('/', methods=['POST', 'OPTIONS']) #this is a route to handle audio transcription and summarization
def transcribe():
    print("\n=== New Request ===")
    print("Request Method:", request.method)
    print("Request Headers:", dict(request.headers))
    print("Request Origin:", request.headers.get('Origin'))

    # Handle preflight requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    print("Received transcription request")
    
    #this if statement checks if the audio file is provided in the request
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({'error': 'No audio file provided'}), 400 

    audio_file = request.files['audio']
    print(f"Received audio file: {audio_file.filename}")

    temp_dir = "static" #string which stores the name of the temporary directory which will store the audio file  

    os.makedirs(temp_dir, exist_ok=True) #this will create the directory if it doesn't exist of name - string stored in temp_dir
    audio_path = os.path.join(temp_dir, audio_file.filename) #this will join the temp_dir and the audio_file.filename and store it in audio_path
    audio_file.save(audio_path) #this will save the audio file to the audio_path
    print(f"Saved audio file to: {audio_path}")

    try:
        print("Processing audio file...")
        result = process_audio(audio_path)
        print("Processing complete. Result:", result)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up: remove the temporary audio file after processing
        os.remove(audio_path)
        print("Cleaned up temporary audio file")

    # Create and configure the response
    response = make_response(jsonify(result))
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Content-Type', 'application/json')
    print("Sending response with headers:", dict(response.headers))
    return response

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000/")
    app.run(debug=True, port=5000)
    
