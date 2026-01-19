import requests
import json
import time

def test_query(prompt):
    print(f"\nQuery: {prompt}")
    url = "http://localhost:8000/api/agent/chat"
    payload = {"message": prompt}
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print(f"Final Answer ({end_time - start_time:.2f}s):")
        print("-" * 50)
        print(data.get("final_answer", "No answer found"))
        print("-" * 50)
        
        # Check if image tags are present
        if "![" in data.get("final_answer", ""):
            print("⚠️ WARNING: Still found images in the final answer!")
        else:
            print("✅ SUCCESS: No images in the final answer (as expected).")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_query("what is the boy 35 evv?")
