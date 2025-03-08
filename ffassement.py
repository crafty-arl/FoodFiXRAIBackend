from pydantic import BaseModel
from typing import Dict, List
from crewai import Agent, Task, Crew, Process
from crewai_tools import TXTSearchTool
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Remove hardcoded API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class AssessmentResponse(BaseModel):
    """Model for individual category assessment response"""
    score: float
    recommendations: List[str]

class AssessmentResult(BaseModel):
    """Model for assessment results"""
    overall_score: float
    categories: Dict[str, AssessmentResponse]
    recommendations: List[str]

class FoodFiXRAssessmentAgent:
    def __init__(self, search_tool: TXTSearchTool):
        self.search_tool = search_tool
        self.question_context = {}  # Store question context
        self.agent = Agent(
            role="FoodFiXR Knowledge Expert",
            goal="Provide accurate nutrition and health information from knowledge base",
            backstory="""I'm a nutrition and health expert that provides evidence-based information 
            from verified knowledge sources. I analyze queries and provide accurate, science-backed 
            responses using only information from the knowledge base.""",
            tools=[search_tool],
            max_iterations=1,
            allow_delegation=False,
            verbose=False
        )

    def analyze_responses(self, responses: Dict[str, float]) -> AssessmentResult:
        """Analyze assessment responses and generate recommendations
        
        Args:
            responses: Dictionary mapping category names to their scores
            
        Returns:
            AssessmentResult containing overall score, category details and recommendations
        """
        # Calculate overall score as average of category scores
        overall_score = sum(responses.values()) / len(responses) if responses else 0
        
        # Process each category
        categories = {}
        all_recommendations = []
        
        # Category descriptions from CSV for context
        category_context = {
            "Toxins": "Focus on reducing exposure to trans fats, excitotoxins, and processed foods while increasing organic and locally sourced options",
            "Sugar": "Address sugar intake and cravings, focusing on low glycemic alternatives and overall sugar reduction",
            "Alkalinity": "Balance body pH through diet choices, emphasizing fresh vegetables and reducing acidic foods",
            "Food Combining": "Optimize nutrient absorption through proper food combinations and timing",
            "Timing": "Establish consistent meal timing patterns for optimal metabolism and digestion",
            "Pre/probiotics": "Support gut health through natural probiotics and prebiotic-rich foods",
            "Macros": "Balance macronutrients with focus on protein intake and healthy fats while managing carbs",
            "Gut/Brain Health": "Support the gut-brain axis through proper nutrition and dietary choices"
        }
        
        for category, score in responses.items():
            # Get question context for this category
            category_questions = "\n".join([
                f"Question {qid}: {q['text']} (Answer: {q['answer']})"
                for qid, q in self.question_context.items()
                if q.get('category', category) == category
            ])
            
            # Create task to analyze this category
            task = Task(
                description=f"""Analyze the score of {score} for category '{category}' and provide recommendations.
                
                Category Context: {category_context.get(category, '')}
                
                Questions and Answers:
                {category_questions}
                
                User Profile: {getattr(self, 'user_profile', {})}
                
                Rules:
                1. Only use information found in the knowledge base
                2. Provide 1-3 specific, actionable recommendations
                3. Keep recommendations clear and concise
                4. Focus on evidence-based improvements
                5. Consider user's dietary preferences and restrictions
                6. Tailor advice to the specific category context and question responses
                
                Format your response as a list with one recommendation per line.
                """,
                expected_output="List of recommendations for the category",
                tools=[self.search_tool],
                agent=self.agent
            )
            
            # Create a crew to execute the task
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            # Get recommendations from knowledge base
            result = crew.kickoff()
            # Convert result to string and split into recommendations
            result_str = str(result)
            # Clean up the recommendations by splitting on newlines and filtering empty lines
            category_recommendations = [
                rec.strip() 
                for rec in result_str.split('\n') 
                if rec.strip()
            ]
            
            # If no recommendations were found, add a default one
            if not category_recommendations:
                category_recommendations = [f"No specific recommendations found for {category}. Please consult with a healthcare professional."]
            
            # Store category results
            categories[category] = AssessmentResponse(
                score=score,
                recommendations=category_recommendations
            )
            
            # Add category recommendations to overall list
            all_recommendations.extend(category_recommendations)
        
        return AssessmentResult(
            overall_score=overall_score,
            categories=categories,
            recommendations=all_recommendations
        )

def process_category_goals(category_data: dict) -> AssessmentResult:
    """Process category-specific assessment data and generate recommendations
    
    Args:
        category_data: Dictionary containing category-specific responses and metadata
        
    Returns:
        AssessmentResult containing category score and recommendations
    """
    # Extract category, responses, and average score
    category = category_data.get("category", "")
    responses = category_data.get("responses", [])
    average_score = category_data.get("averageScore", 0)  # Use provided average score
    
    # Initialize the search tool and agent
    search_tool = TXTSearchTool()
    agent = FoodFiXRAssessmentAgent(search_tool)
    
    # Add user profile if provided
    if "userProfile" in category_data:
        agent.user_profile = category_data["userProfile"]
    
    # Create a dictionary with just this category's score
    category_scores = {category: average_score}
    
    # Add question context to the agent
    question_context = {
        response["questionId"]: {
            "text": response["questionText"],
            "answer": response["answer"],
            "category": response.get("category", category)
        }
        for response in responses
    }
    agent.question_context = question_context
    
    # Analyze responses and generate recommendations
    assessment_result = agent.analyze_responses(category_scores)
    
    return assessment_result

def format_assessment_result(result: AssessmentResult) -> dict:
    """Format the assessment result into a JSON structure
    
    Args:
        result: AssessmentResult object to format
        
    Returns:
        Dictionary containing formatted assessment results
    """
    formatted_result = {
        "overall_score": round(result.overall_score, 1),
        "categories": {}
    }
    
    for category, response in result.categories.items():
        # Filter out any recommendations that end with markdown artifacts
        clean_recommendations = [
            rec.strip() 
            for rec in response.recommendations 
            if not rec.endswith('```')
        ]
        
        # Remove numbered prefixes if they exist
        clean_recommendations = [
            rec[rec.find('.') + 1:].strip() if rec[0].isdigit() else rec
            for rec in clean_recommendations
        ]
        
        formatted_result["categories"][category] = {
            "score": round(response.score, 1),
            "recommendations": clean_recommendations
        }
    
    return formatted_result


