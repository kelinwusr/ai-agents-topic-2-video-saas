import json
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_openai_client():
    """
    Initialize and return the OpenAI client with proper error handling.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found! Please set your OPENAI_API_KEY environment variable.\n"
            "You can do this by either:\n"
            "1. Creating a .env file with OPENAI_API_KEY=your-key-here\n"
            "2. Setting it in your terminal: export OPENAI_API_KEY=your-key-here"
        )
    
    return OpenAI(api_key=api_key)

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

def generate_video_scenes(topic):
    """
    Generate video scenes context based on the given topic using OpenAI's language model.
    :param topic: The topic for which video scenes context needs to be generated.
    :return: A JSON object containing video scenes context.
    """
    try:
        client = get_openai_client()
        
        prompt = f"Create a detailed video scene breakdown for a video about {topic.title}. Include different scenes, key elements, and descriptions."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        scene_context = response.choices[0].message.content.strip()
        # Convert scene context to JSON object
        scenes = [scene.strip() for scene in scene_context.split("\n") if scene]
        json_output = json.dumps({"topic": topic, "scenes": scenes}, indent=4)
        return json_output
    
    except ValueError as ve:
        return json.dumps({"error": str(ve)})
    except Exception as e:
        return json.dumps({"error": f"An error occurred: {str(e)}"})

def main():
    if len(sys.argv) != 2:
        print("Usage: python crew.py <topic>")
        sys.exit(1)
    
    topic = sys.argv[1]
    video_scenes = generate_video_scenes(topic)
    print(video_scenes)

if __name__ == "__main__":
    main()