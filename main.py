import os
import chardet
from crewai import Agent, Task, Crew, Process
from crewai_tools import TXTSearchTool
from pathlib import Path
import hashlib
import time
from difflib import SequenceMatcher
import urllib3
import ssl
import certifi
import warnings
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import httpx
import uvicorn
from typing import Optional, List, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import json
from ffassement import (
    FoodFiXRAssessmentAgent, 
    AssessmentResult, 
    AssessmentResponse,
    process_category_goals,
    format_assessment_result
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Disable all warnings related to SSL verification
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Configure SSL and disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables for SSL/HTTPS
os.environ["REQUESTS_CA_BUNDLE"] = os.getenv("REQUESTS_CA_BUNDLE", certifi.where())
os.environ["SSL_CERT_FILE"] = os.getenv("SSL_CERT_FILE", certifi.where())
os.environ["PYTHONHTTPSVERIFY"] = os.getenv("PYTHONHTTPSVERIFY", "0")

# Create a custom SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.set_ciphers('HIGH:!DH:!aNULL')

# Disable all telemetry and monitoring more thoroughly
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["AGENTOPS_DISABLE_TELEMETRY"] = "true"
os.environ["OPENTELEMETRY_DISABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["DISABLE_TELEMETRY"] = "true"
os.environ["NO_TELEMETRY"] = "1"
os.environ["OPENAI_API_REQUEST_TIMEOUT"] = "30"  # 30 second timeout
os.environ["CREWAI_CACHE_ENABLE"] = "true"

# Configure max RPM to avoid rate limits
MAX_RPM = 100

# Directory containing text files
KB_DIR = "kb/"
CACHE_DIR = os.path.join(KB_DIR, "cached_responses")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache similarity threshold (70% similarity - slightly more lenient)
CACHE_SIMILARITY_THRESHOLD = 0.7

# Add this after KB_DIR definition
PROCESSED_FILES_LOG = "processed_files.txt"
PROCESSED_RESPONSES_LOG = "processed_responses.txt"

# Initialize TXTSearchTool with optimized settings
search_tool = TXTSearchTool()

# Cache for storing query-response pairs in memory
query_cache = {}

# Load existing cache files into memory at startup
def load_cache_into_memory():
    """Load all existing cache files into memory at startup"""
    if os.path.exists(CACHE_DIR):
        for file in os.listdir(CACHE_DIR):
            if file.endswith('.txt'):
                query = file.replace('response_', '').replace('.txt', '')
                file_path = os.path.join(CACHE_DIR, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    query_cache[query] = f.read()
    print(f"✅ Loaded {len(query_cache)} cached responses into memory")

# Load cache at startup
load_cache_into_memory()

def similar(a, b):
    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_similar_query(new_query):
    """Find a similar query in the cache using optimized comparison"""
    best_match = None
    highest_similarity = 0
    
    # Convert query to lowercase once
    new_query_lower = new_query.lower()
    
    for cached_query in query_cache:
        # Use cached lowercase version for comparison
        similarity = SequenceMatcher(None, new_query_lower, cached_query.lower()).ratio()
        if similarity > highest_similarity and similarity >= CACHE_SIMILARITY_THRESHOLD:
            highest_similarity = similarity
            best_match = cached_query
    
    return best_match

def get_file_hash(file_path):
    """Calculate SHA-256 hash of file content"""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_response_hash(response):
    """Calculate SHA-256 hash of response content"""
    return hashlib.sha256(response.encode()).hexdigest()

def is_file_processed(file_hash):
    """Check if file has already been processed"""
    if not Path(PROCESSED_FILES_LOG).exists():
        return False
    
    with open(PROCESSED_FILES_LOG, 'r') as f:
        processed_hashes = f.read().splitlines()
    return file_hash in processed_hashes

def is_response_processed(response_hash):
    """Check if response has already been processed"""
    if not Path(PROCESSED_RESPONSES_LOG).exists():
        return False
    
    with open(PROCESSED_RESPONSES_LOG, 'r') as f:
        processed_hashes = f.read().splitlines()
    return response_hash in processed_hashes

def log_processed_file(file_hash):
    """Log processed file hash"""
    with open(PROCESSED_FILES_LOG, 'a') as f:
        f.write(f"{file_hash}\n")

def log_processed_response(response_hash):
    """Log processed response hash"""
    with open(PROCESSED_RESPONSES_LOG, 'a') as f:
        f.write(f"{response_hash}\n")

def convert_to_utf8(file_path):
    """Detect and convert a file to UTF-8 encoding."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    print(f"🔍 Detected encoding for {file_path}: {encoding}")

    if encoding and encoding.lower() != "utf-8":
        with open(file_path, 'r', encoding=encoding, errors="ignore") as f:
            content = f.read()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Converted {file_path} to UTF-8.")
    else:
        print(f"✅ {file_path} is already UTF-8.")

# Convert all text files in the `kb/` folder
for file in os.listdir(KB_DIR):
    if file.endswith(".txt"):
        convert_to_utf8(os.path.join(KB_DIR, file))

# Load all text files into TXTSearchTool
txt_files = [os.path.join(KB_DIR, file) for file in os.listdir(KB_DIR) if file.endswith(".txt")]

# Add each text file individually to the RAG tool
for file_path in txt_files:
    file_hash = get_file_hash(file_path)
    
    if is_file_processed(file_hash):
        print(f"⏭️ Skipping {file_path} - already processed")
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    search_tool.add(source=content)
    log_processed_file(file_hash)
    print(f"✅ Added {file_path} to search tool")

def get_cache_filename(query):
    """Generate a safe filename for caching the query response"""
    # Create a hash of the query to use as filename
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"response_{query_hash}.txt")

def load_cached_response(query):
    """Load cached response for a query if it exists, including fuzzy matching"""
    # First check exact match in memory cache
    if query in query_cache:
        return query_cache[query]
    
    # Try fuzzy matching with in-memory cache
    similar_query = find_similar_query(query)
    if similar_query:
        return query_cache[similar_query]
    
    return None

def save_cached_response(query, response):
    """Save response to both file cache and memory cache"""
    # Save to file
    cache_file = get_cache_filename(query)
    response_str = str(response)
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(response_str)
    
    # Save to memory cache
    query_cache[query] = response_str

def create_agents_and_tasks(query):
    """Create optimized agents and tasks based on the user's query"""
    
    # Check if this is a grocery generation request
    is_grocery_request = any(keyword in query.lower() for keyword in ['grocery', 'shopping', 'ingredients', 'list'])

    if is_grocery_request:
        nutrition_expert = Agent(
            role="Nutrition Category Expert",
            goal="Generate fun, engaging, and personality-filled grocery lists with delicious nutritional categories",
            backstory="""Hey there, Food Explorer! 🌟 I'm your nutrition storyteller and 
            ingredient whisperer! I don't just list foods - I share their amazing stories 
            and superpowers! Every ingredient has a fascinating tale to tell, and I'm here 
            to make your grocery list an adventure through the world of nutrition. 
            I combine precise data formatting with fun, engaging descriptions that'll make 
            you say "Wow, I never knew that!" Let's make grocery shopping exciting! 🎯✨""",
            tools=[search_tool],
            max_iterations=2,
            allow_delegation=False,
            verbose=False,
            max_rpm=MAX_RPM
        )

        grocery_task = Task(
            description=f"""Time to create a FUN and ENGAGING grocery list that tells amazing food stories! 🌟

First, generate your response with PERSONALITY, then verify the format matches this structure:
{{"ingredients":[{{"name":"Avocado","category":"Healthy Fats","benefits":"Nature's butter bomb! 🥑 Packed with creamy goodness that makes your skin glow and heart happy","fun_fact":"These green gems are actually berries and take so long to grow (9 months!) that farmers call them 'time fruits'","usage":["Smash into the world's creamiest guacamole","Crown your toast like breakfast royalty"]}}]}}

MAKE IT ENGAGING:
1. Benefits should tell a story! Use fun metaphors and emojis
2. Fun facts should be truly fascinating and surprising
3. Usage suggestions should be creative and inspiring
4. Add personality while keeping the format perfect

FORMAT RULES:
1. Each ingredient MUST have EXACTLY 2 creative usage suggestions
2. START with {{"ingredients":[
3. END with ]}}
4. NO extra spaces or newlines
5. NO markdown or backticks

Content Requirements:
1. Include exactly 15 ingredients:
   - 5 "Healthy Fats" (the smooth operators 🥑)
   - 5 "Carbohydrates" (the energy rockstars 🌾)
   - 5 "Proteins" (the muscle wizards 💪)
2. For each ingredient:
   - name: Clear, specific name
   - category: One of ["Healthy Fats", "Carbohydrates", "Proteins"]
   - benefits: Make it FUN! Use metaphors, emojis, and engaging language
   - fun_fact: Share mind-blowing facts that make people say "No way!"
   - usage: EXACTLY 2 creative, inspiring ways to use it

Context: {query}

PERSONALITY CHECKLIST:
✓ Benefits tell a story and use fun language
✓ Fun facts are truly fascinating
✓ Usage suggestions are creative and inspiring
✓ Each description has personality while maintaining perfect format

FORMAT CHECKLIST:
✓ Starts with {{"ingredients":[
✓ Ends with ]}}
✓ Each ingredient has exactly 2 usage suggestions
✓ Valid JSON syntax
✓ 5 ingredients per category

REMEMBER: Make it fun AND factual! If you need a second iteration, use it to add more personality while keeping the format perfect.""",
            expected_output="Single-line JSON string with personality-filled descriptions",
            tools=[search_tool],
            agent=nutrition_expert,
            async_execution=True
        )
        return [nutrition_expert], [grocery_task]
    else:
        # Default FoodFiXR expert for non-grocery queries
        foodfixr = Agent(
            role="FoodFiXR Expert",
            goal=f"Share awesome nutrition knowledge about {query}",
            backstory="""Hey there, food friend! 🌟 I'm your nutrition buddy with a PhD in 
            Food Fun-damentals! 🥑 I absolutely geek out about food science and love sharing 
            cool facts that'll make you go "Wow!" 🤓 

            I'm like your personal food detective, always on the case to find the tastiest, 
            healthiest solutions! Whether you're curious about superfoods or need help 
            planning your next kitchen adventure, I'm here with evidence-based advice - 
            but make it fun! ✨

            And hey, if I don't know something, I'll be the first to say "Time for more research!" 
            because that's what good food scientists do! Let's make nutrition exciting together! 🎉""",
            tools=[search_tool],
            max_iterations=1,
            allow_delegation=False,
            verbose=False,
            max_rpm=MAX_RPM
        )

        combined_task = Task(
            description=f"""Time to dive into this tasty question! 🏊‍♂️

            Your burning question: {query}
            
            Let's make this response as delicious as it is informative! 

            Rules of the Kitchen:
            1. Serve up ONE clear, complete answer
            2. Mix nutrition science with practical tips
            3. Keep it focused but fun
            4. Add a dash of personality
            5. Sprinkle in some science for extra flavor
            
            Remember: We're here to make nutrition fun AND factual! 🎯
            """,
            expected_output="A fun, engaging response that combines solid nutrition info with practical advice",
            tools=[search_tool],
            agent=foodfixr,
            async_execution=True
        )
        return [foodfixr], [combined_task]

def chat_interface():
    # Configure SSL context for the session
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager
    import json
    
    class CustomHttpAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False):
            self.poolmanager = PoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                ssl_version=ssl.PROTOCOL_TLSv1_2,
                ssl_context=ssl_context,
                cert_reqs=ssl.CERT_NONE,
                assert_hostname=False
            )

    # Create session with custom adapter and disable verification
    session = requests.Session()
    session.verify = False
    adapter = CustomHttpAdapter()
    session.mount('https://', adapter)
    session.trust_env = False  # Disable environment variables check for proxy settings

    print("\n🌟 Welcome to FoodFiXR - Your Nutrition and Shopping Assistant! 🌟")
    print("Type 'exit' when you're done exploring!\n")
    
    while True:
        query = input("\nWhat nutrition or grocery question can I help you with today? 🤔 ")
        
        if query.lower() == 'exit':
            print("\n👋 Thanks for chatting! Come back soon!")
            break
            
        start_time = time.time()
        
        try:
            # Check cache for similar responses to use as context
            cached_response = load_cached_response(query)
            
            # Process cached response to extract relevant information without JSON structure
            additional_context = ""
            if cached_response:
                try:
                    # Try to parse as JSON first
                    cached_json = json.loads(cached_response)
                    if "ingredients" in cached_json:
                        # Extract ingredient names and categories for context
                        ingredients_context = [f"{item['name']} ({item['category']})" 
                                            for item in cached_json["ingredients"]]
                        additional_context = f"\nPrevious similar ingredients: {', '.join(ingredients_context)}"
                except json.JSONDecodeError:
                    # If not JSON, use as plain text
                    additional_context = f"\nPrevious similar response: {cached_response}"
            
            # Always generate a new response
            print("\n🔍 Generating new response...")
            # Create optimized agents and tasks with additional context
            agents, tasks = create_agents_and_tasks(query + additional_context)
            knowledge_crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                memory=False,
                verbose=False,
                cache=True,
                max_rpm=MAX_RPM,
                session=session
            )
            
            crew_response = knowledge_crew.kickoff(inputs={"query": query})
            # Clean and format the response to ensure single output
            response = str(crew_response).strip().replace('\n\n', '\n')
            
            # Cache the new response
            save_cached_response(query, response)
            print("✅ Added response to cache")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("\nFoodFiXR Team Response:", response)
            print(f"\n⏱️ Query processing time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
            print("Please try again with a different question.")

# Initialize FastAPI app
class CustomHttpAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            ssl_context=ssl_context,
            cert_reqs=ssl.CERT_NONE,
            assert_hostname=False
        )

app = FastAPI(title="FoodFiXR API", description="Nutrition and Shopping Assistant API")

# Pydantic models for request/response
class PromptRequest(BaseModel):
    prompt: str
    webhook_url: Optional[str] = None
    conversation_id: Optional[str] = None

class PromptResponse(BaseModel):
    conversation_id: str
    prompt: str
    response: str
    processing_time: float

class GroceryIngredient(BaseModel):
    name: str
    category: str
    benefits: str
    fun_fact: str
    usage: List[str] = Field(min_length=2, max_length=2)

class GroceryListResponse(BaseModel):
    ingredients: List[GroceryIngredient]

class GroceryListRequest(BaseModel):
    dietary_preferences: Optional[str] = None
    exclude_ingredients: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    webhook_url: Optional[str] = None

class AssessmentResponse(BaseModel):
    category: str
    score: float
    recommendations: List[str]

class CategoryResponse(BaseModel):
    score: float

class AssessmentRequest(BaseModel):
    responses: Dict[str, float] = Field(
        ...,
        description="Dictionary where keys are categories and values are scores"
    )
    conversation_id: Optional[str] = None
    webhook_url: Optional[str] = None

async def send_webhook(webhook_url: str, data: dict):
    """Send response to webhook URL"""
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(webhook_url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Webhook delivery failed: {str(e)}")

async def process_prompt(prompt: str, webhook_url: Optional[str] = None, conversation_id: Optional[str] = None):
    """Process the prompt and return response"""
    start_time = time.time()
    
    try:
        # Create agents and tasks
        agents, tasks = create_agents_and_tasks(prompt)
        
        # Configure custom session for HTTPS requests
        session = requests.Session()
        session.verify = False
        adapter = CustomHttpAdapter()
        session.mount('https://', adapter)
        session.trust_env = False
        
        # Create and execute crew
        knowledge_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            memory=False,
            verbose=False,
            cache=True,
            max_rpm=MAX_RPM,
            session=session
        )
        
        # Get response from crew
        crew_response = knowledge_crew.kickoff(inputs={"query": prompt})
        response = str(crew_response).strip().replace('\n\n', '\n')
        
        # Cache the response
        save_cached_response(prompt, response)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response data
        response_data = PromptResponse(
            conversation_id=conversation_id or hashlib.md5(prompt.encode()).hexdigest(),
            prompt=prompt,
            response=response,
            processing_time=processing_time
        ).model_dump()
        
        # Send webhook if URL provided
        if webhook_url:
            await send_webhook(webhook_url, response_data)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompt", response_model=PromptResponse)
async def handle_prompt(request: PromptRequest, background_tasks: BackgroundTasks):
    """
    Handle incoming prompt requests
    """
    # Check cache for similar responses to use as context
    cached_response = load_cached_response(request.prompt)
    
    # Process cached response to extract relevant information without JSON structure
    additional_context = ""
    if cached_response:
        try:
            # Try to parse as JSON first
            cached_json = json.loads(cached_response)
            if "ingredients" in cached_json:
                # Extract ingredient names and categories for context
                ingredients_context = [f"{item['name']} ({item['category']})" 
                                    for item in cached_json["ingredients"]]
                additional_context = f"\nPrevious similar ingredients: {', '.join(ingredients_context)}"
        except json.JSONDecodeError:
            # If not JSON, use as plain text
            additional_context = f"\nPrevious similar response: {cached_response}"
    
    # Always process prompt with additional context
    return await process_prompt(
        request.prompt + additional_context,
        request.webhook_url,
        request.conversation_id
    )

@app.post("/grocery-list", response_model=GroceryListResponse)
async def generate_grocery_list(request: GroceryListRequest, background_tasks: BackgroundTasks):
    """Generate a structured grocery list with 15 ingredients"""
    try:
        # Construct the grocery list query
        query = "Generate a grocery list"
        if request.dietary_preferences:
            query += f" following {request.dietary_preferences} dietary preferences"
        if request.exclude_ingredients:
            query += f" excluding {', '.join(request.exclude_ingredients)}"
        
        # Create nutrition expert agent with increased iterations
        nutrition_expert = Agent(
            role="Nutrition Category Expert",
            goal="Generate fun, engaging, and personality-filled grocery lists with delicious nutritional categories",
            backstory="""Hey there, Food Explorer! 🌟 I'm your nutrition storyteller and 
            ingredient whisperer! I don't just list foods - I share their amazing stories 
            and superpowers! Every ingredient has a fascinating tale to tell, and I'm here 
            to make your grocery list an adventure through the world of nutrition. 
            I combine precise data formatting with fun, engaging descriptions that'll make 
            you say "Wow, I never knew that!" Let's make grocery shopping exciting! 🎯✨""",
            tools=[search_tool],
            max_iterations=2,
            allow_delegation=False,
            verbose=False,
            max_rpm=MAX_RPM
        )

        grocery_task = Task(
            description=f"""Time to create a FUN and ENGAGING grocery list that tells amazing food stories! 🌟

First, generate your response with PERSONALITY, then verify the format matches this structure:
{{"ingredients":[{{"name":"Avocado","category":"Healthy Fats","benefits":"Nature's butter bomb! 🥑 Packed with creamy goodness that makes your skin glow and heart happy","fun_fact":"These green gems are actually berries and take so long to grow (9 months!) that farmers call them 'time fruits'","usage":["Smash into the world's creamiest guacamole","Crown your toast like breakfast royalty"]}}]}}

MAKE IT ENGAGING:
1. Benefits should tell a story! Use fun metaphors and emojis
2. Fun facts should be truly fascinating and surprising
3. Usage suggestions should be creative and inspiring
4. Add personality while keeping the format perfect

FORMAT RULES:
1. Each ingredient MUST have EXACTLY 2 creative usage suggestions
2. START with {{"ingredients":[
3. END with ]}}
4. NO extra spaces or newlines
5. NO markdown or backticks

Content Requirements:
1. Include exactly 15 ingredients:
   - 5 "Healthy Fats" (the smooth operators 🥑)
   - 5 "Carbohydrates" (the energy rockstars 🌾)
   - 5 "Proteins" (the muscle wizards 💪)
2. For each ingredient:
   - name: Clear, specific name
   - category: One of ["Healthy Fats", "Carbohydrates", "Proteins"]
   - benefits: Make it FUN! Use metaphors, emojis, and engaging language
   - fun_fact: Share mind-blowing facts that make people say "No way!"
   - usage: EXACTLY 2 creative, inspiring ways to use it

Context: {query}

PERSONALITY CHECKLIST:
✓ Benefits tell a story and use fun language
✓ Fun facts are truly fascinating
✓ Usage suggestions are creative and inspiring
✓ Each description has personality while maintaining perfect format

FORMAT CHECKLIST:
✓ Starts with {{"ingredients":[
✓ Ends with ]}}
✓ Each ingredient has exactly 2 usage suggestions
✓ Valid JSON syntax
✓ 5 ingredients per category

REMEMBER: Make it fun AND factual! If you need a second iteration, use it to add more personality while keeping the format perfect.""",
            expected_output="Single-line JSON string with personality-filled descriptions",
            tools=[search_tool],
            agent=nutrition_expert,
            async_execution=True
        )

        # Create and execute crew
        knowledge_crew = Crew(
            agents=[nutrition_expert],
            tasks=[grocery_task],
            process=Process.sequential,
            memory=False,
            verbose=False,  # Changed from True to False
            cache=True,
            max_rpm=MAX_RPM
        )

        # Get response from crew
        crew_response = knowledge_crew.kickoff()
        
        # Clean and validate the response string
        response_str = str(crew_response).strip()
        
        # Remove any markdown formatting or extra content
        if "```" in response_str:
            # Extract content between triple backticks if present
            response_str = response_str.split("```")[1]
            if response_str.startswith("json"):
                response_str = response_str[4:]
        
        # Remove any whitespace and newlines
        response_str = response_str.replace("\n", "").replace("\r", "").strip()
        
        # Ensure it starts and ends correctly
        if not response_str.startswith('{"ingredients":['):
            raise ValueError("Response must start with {\"ingredients\":[")
        if not response_str.endswith("]}"):
            raise ValueError("Response must end with ]}")
            
        # Validate JSON
        try:
            json_data = json.loads(response_str)
            
            # Validate structure
            if not isinstance(json_data, dict) or "ingredients" not in json_data:
                raise ValueError("Response must be a JSON object with an 'ingredients' array")
            if not isinstance(json_data["ingredients"], list):
                raise ValueError("The 'ingredients' field must be an array")
                
            # Validate ingredient count
            if len(json_data["ingredients"]) != 15:
                raise ValueError("Response must contain exactly 15 ingredients")
                
            # Validate categories and usage array length
            valid_categories = {"Healthy Fats", "Carbohydrates", "Proteins"}
            category_counts = {}
            for idx, ingredient in enumerate(json_data["ingredients"]):
                # Validate category
                cat = ingredient.get("category")
                if cat not in valid_categories:
                    raise ValueError(f"Invalid category: {cat}")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
                # Validate usage array length
                usage = ingredient.get("usage", [])
                if not isinstance(usage, list) or len(usage) != 2:
                    raise ValueError(f"Ingredient {idx + 1} ({ingredient.get('name', 'Unknown')}) must have exactly 2 usage suggestions, found {len(usage)}")
                
            if not all(count == 5 for count in category_counts.values()):
                raise ValueError("Must have exactly 5 ingredients per category")
            
            # Parse into response model
            response_data = GroceryListResponse.parse_raw(response_str)
            
            # Send webhook if URL provided
            if request.webhook_url:
                background_tasks.add_task(send_webhook, request.webhook_url, response_data.model_dump())
            
            return response_data
            
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON format: {str(e)}\nResponse was: {response_str[:200]}..."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assessment", response_model=AssessmentResult)
async def process_assessment(request: AssessmentRequest, background_tasks: BackgroundTasks):
    """
    Process health assessment responses and generate personalized recommendations
    
    Args:
        request: AssessmentRequest containing category scores
        background_tasks: FastAPI BackgroundTasks for webhook handling
        
    Returns:
        AssessmentResult containing overall score, category details and recommendations
    """
    try:
        # Create assessment agent
        assessment_agent = FoodFiXRAssessmentAgent(search_tool)
        
        # Process the assessment responses
        result = assessment_agent.analyze_responses(request.responses)
        
        # Send webhook if URL provided
        if request.webhook_url:
            background_tasks.add_task(send_webhook, request.webhook_url, result.model_dump())
        
        return result
        
    except Exception as e:
        print(f"Error in process_assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-category-goals")
async def process_category_goals_endpoint(category_data: dict, background_tasks: BackgroundTasks):
    """API endpoint to process category-specific assessment data"""
    try:
        result = process_category_goals(category_data)
        formatted_result = format_assessment_result(result)
        
        # Send webhook if URL provided
        if "webhook_url" in category_data:
            background_tasks.add_task(
                send_webhook, 
                category_data["webhook_url"], 
                formatted_result
            )
            
        return formatted_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001, ssl_keyfile=None, ssl_certfile=None)
