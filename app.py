import json
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CrewBase:
    """
    Base class for crew-related tasks.
    """
    @staticmethod
    def agent():
        """
        Represents an agent within the crew.
        """
        return "Agent initialized"

    @staticmethod
    def task():
        """
        Represents a task within the crew.
        """
        return "Task initialized"

    @staticmethod
    def crew():
        """
        Represents the crew information.
        """
        return "Crew initialized"

def generate_video_scenes(topic: str) -> dict:
    """
    Generate video scenes context based on the given topic using OpenAI's language model.
    
    Args:
        topic (str): The topic for which video scenes context needs to be generated.
    
    Returns:
        dict: A dictionary containing video scenes context.
    
    Raises:
        Exception: If there's an error in generating the scenes.
    """
    try:
        # Create the messages array for chat completion
        messages = [
            {"role": "system", "content": "You are a professional video scene writer."},
            {"role": "user", "content": f"""Create a detailed video scene breakdown for a video about {topic}. 
             Format the output as a list of scenes, each with a scene number, description, and key elements.
             Make it practical and filmable."""}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the content from the response
        scene_content = response.choices[0].message.content.strip()

        # Parse the scenes into a structured format
        scenes = []
        current_scene = {}
        
        for line in scene_content.split('\n'):
            line = line.strip()
            if line:
                if line.lower().startswith('scene'):
                    if current_scene:
                        scenes.append(current_scene)
                    current_scene = {'number': line, 'description': '', 'elements': []}
                elif current_scene:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        if key.lower().strip() == 'description':
                            current_scene['description'] = value.strip()
                        elif key.lower().strip() == 'key elements':
                            current_scene['elements'] = [elem.strip() for elem in value.split(',')]
                    else:
                        if not current_scene['description']:
                            current_scene['description'] = line

        if current_scene:
            scenes.append(current_scene)

        return {
            "status": "success",
            "topic": topic,
            "scenes": scenes
        }

    except Exception as e:
        logger.error(f"Error generating video scenes: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def main(topic: str) -> dict:
    """
    Main function to handle video scene generation.
    
    Args:
        topic (str): The topic for scene generation.
    
    Returns:
        dict: The generated scenes or error message.
    """
    try:
        return generate_video_scenes(topic)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return {"status": "error", "message": str(e)}

# Flask API setup
app = Flask(__name__)
CORS(app)

@app.route('/api/crewai', methods=['POST'])
def execute_crewai():
    """
    API endpoint to handle video scene generation requests.
    """
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        
        # Validate request data
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        topic = data.get('topic')
        if not topic:
            return jsonify({
                'status': 'error',
                'message': 'Topic is required'
            }), 400

        # Generate video scenes
        result = main(topic)
        
        if result.get('status') == 'error':
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Validate environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set in environment variables")
        sys.exit(1)
        
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)