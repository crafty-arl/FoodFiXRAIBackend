import os
import pytest
import json
from fastapi.testclient import TestClient
from main import app, create_agents_and_tasks, load_cached_response, save_cached_response

# Create test client
client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    print("\nHealth Check Test Results:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_prompt_endpoint():
    """Test the /prompt endpoint"""
    test_prompt = "What are healthy breakfast options?"
    request_data = {
        "prompt": test_prompt,
        "conversation_id": "test-123"
    }
    
    response = client.post("/prompt", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "conversation_id" in data
    assert "prompt" in data
    assert "response" in data
    assert "processing_time" in data
    assert data["prompt"] == test_prompt
    print("\nPrompt Endpoint Test Results:")
    print(f"Status Code: {response.status_code}")
    print(f"Response Data: {json.dumps(data, indent=2)}")

def test_grocery_list_endpoint():
    """Test the /grocery-list endpoint"""
    request_data = {
        "dietary_preferences": "vegetarian",
        "exclude_ingredients": ["nuts"],
        "conversation_id": "test-123"
    }
    
    response = client.post("/grocery-list", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "ingredients" in data
    assert len(data["ingredients"]) == 15
    
    # Verify ingredient categories
    categories = {"Healthy Fats": 0, "Carbohydrates": 0, "Proteins": 0}
    for ingredient in data["ingredients"]:
        categories[ingredient["category"]] += 1
    
    assert categories["Healthy Fats"] == 5
    assert categories["Carbohydrates"] == 5
    assert categories["Proteins"] == 5
    print("\nGrocery List Endpoint Test Results:")
    print(f"Status Code: {response.status_code}")
    print(f"Number of Ingredients: {len(data['ingredients'])}")
    print(f"Category Distribution: {categories}")
    print(f"Response Data: {json.dumps(data, indent=2)}")

def test_assessment_endpoint():
    """Test the /assessment endpoint"""
    test_responses = {
        "How often do you experience digestive issues?": "never",
        "How often do you consume added sugars?": "rarely"
    }
    
    request_data = {
        "responses": test_responses,
        "conversation_id": "test-123"
    }
    
    response = client.post("/assessment", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "overall_score" in data
    assert "category_scores" in data
    print("\nAssessment Endpoint Test Results:")
    print(f"Status Code: {response.status_code}")
    print(f"Response Data: {json.dumps(data, indent=2)}")

def test_create_agents_and_tasks():
    """Test agent and task creation"""
    # Test grocery list query
    agents, tasks = create_agents_and_tasks("Create a grocery list for a vegan diet")
    assert len(agents) == 1
    assert len(tasks) == 1
    assert agents[0].role == "Nutrition Category Expert"
    print("\nAgent Creation Test Results (Grocery List):")
    print(f"Number of Agents: {len(agents)}")
    print(f"Number of Tasks: {len(tasks)}")
    print(f"Agent Role: {agents[0].role}")
    
    # Test general nutrition query
    agents, tasks = create_agents_and_tasks("What are good sources of protein?")
    assert len(agents) == 1
    assert len(tasks) == 1
    assert agents[0].role == "FoodFiXR Expert"
    print("\nAgent Creation Test Results (Nutrition Query):")
    print(f"Number of Agents: {len(agents)}")
    print(f"Number of Tasks: {len(tasks)}")
    print(f"Agent Role: {agents[0].role}")

def test_cache_functionality():
    """Test response caching"""
    test_query = "What are healthy snacks?"
    test_response = "Here are some healthy snacks: carrots, hummus, nuts"
    
    # Test saving to cache
    save_cached_response(test_query, test_response)
    
    # Test loading from cache
    cached = load_cached_response(test_query)
    assert cached == test_response
    
    # Test similar query matching
    similar_query = "What are some healthy snack options?"
    similar_cached = load_cached_response(similar_query)
    assert similar_cached == test_response
    print("\nCache Functionality Test Results:")
    print(f"Original Query: {test_query}")
    print(f"Cached Response: {cached}")
    print(f"Similar Query: {similar_query}")
    print(f"Similar Query Response: {similar_cached}")

if __name__ == "__main__":
    pytest.main([__file__])