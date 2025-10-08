from crewai import Crew, Process
from tasks import (
    get_routing_task, get_text_analysis_task, get_charting_tasks,
    get_gmail_task, get_hybrid_task
)
from agents import (
    get_router_agent, get_text_analyst_agent, get_data_analyst_agent,
    get_data_preparation_agent, get_code_generation_agent, get_code_execution_agent,
    get_gmail_agent, get_hybrid_agent,get_comparison_agent
)


def create_routing_crew(user_query):
    routing_task = get_routing_task(user_query)
    return Crew(agents=[routing_task.agent],  # Access the agent directly
                tasks=[routing_task],
                process=Process.sequential,
                verbose=True)
        

def create_text_analysis_crew(user_query, retrieved_context):
    text_analysis_task = get_text_analysis_task(user_query, retrieved_context)
    return  Crew(agents=[text_analysis_task.agent],
                        tasks=[text_analysis_task],
                        process=Process.sequential,
                        verbose=True,
                        memory=False  # Ensure context is passed explicitly
                    )

def create_charting_crew(user_query, retrieved_context):
    charting_tasks = get_charting_tasks(user_query, retrieved_context)

    return Crew(
        agents=[task.agent for task in charting_tasks],
        tasks=charting_tasks,
        process=Process.sequential,
        verbose=True
    )

def create_gmail_crew(user_query):
    gmail_task = get_gmail_task(user_query)
    return Crew(agents=[gmail_task.agent],
                    tasks=[gmail_task],
                    process=Process.sequential,
                    verbose=True
                )

def create_hybrid_crew(user_query):
    hybrid_task = get_hybrid_task(user_query)
    return Crew(agents=[hybrid_task.agent],
                    tasks=[hybrid_task],
                    process=Process.sequential,
                    verbose=True
                )