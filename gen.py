import os  
from openai import OpenAI  
import json

# 建议使用环境变量来管理API密钥，避免硬编码  
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  
# 为方便演示，此处直接提供一个虚拟key  
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1") # 替换为你的API Key和Base URL

def generate_similar_queries(intent_name, intent_description, seed_queries, count=10):
    """ 使用LLM为一个意图生成多样化的同义查询。

    Args:  
        intent_name (str): 意图名称。  
        intent_description (str): 意图的详细描述。  
        seed_queries (list): 种子查询列表作为示例。  
        count (int): 希望生成的查询数量。

    Returns:  
        list: 生成的同义查询列表。  
    """      
    # 构建一个强大的提示词，引导LLM进行高质量的生成  
    prompt = f"""  
    你是一个AI智能体的数据增强专家。你的任务是为一个特定的意图生成多样化的用户查询。

    **意图名称:** {intent_name}
    **意图描述:** {intent_description}

    **请参考以下示例查询:**  
    {', '.join(seed_queries)}

    **要求:**  
    1. 生成 {count} 条与上述意图相关、但表达方式不同的用户查询。  
    2. 风格需要口语化、简洁，并模拟真实用户的提问习惯。  
    3. 覆盖不同的句式，例如：陈述句、疑问句、甚至省略部分信息的短语。  
    4. 不要包含礼貌用语，如“请”、“谢谢”。  
    5. 仅输出一个JSON格式的列表，不要包含任何其他解释性文字。例如：["查询1", "查询2", ...]  
    """

    print("--- Sending Prompt to LLM ---")  
    print(prompt)  
    print("-----------------------------")

    try:  
        response = client.chat.completions.create(  
            model="qwen2.5vl",  # 可以替换为你选择的模型  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.8, # 提高一点温度以增加多样性  
            response_format={"type": "json_object"},  
        ) 

        print("--- LLM原始输出 ---")
        print(response)
        print("-------------------") 

        # 假设模型会返回一个包含 "queries" 键的JSON对象  
        generated_text = response.choices[0].message.content  

        # 由于 response_format="json_object"，可以直接解析  
        # 但为了稳健，我们还是做一下检查  
        result_data = json.loads(generated_text) 

        # 假设返回的JSON结构是 {"queries": ["...", "..."]}  
        # 如果不是，你可能需要根据实际返回调整这里的解析逻辑  
        if"queries"in result_data and isinstance(result_data["queries"], list):  
            return result_data["queries"]  
        else:  
            # 尝试直接解析列表  
            return json.loads(generated_text)

    except Exception as e:  
        print(f"An error occurred: {e}")  
        return []

# --- 示例 ---  
# intent_name = "查询公交线路"
# intent_description = "用户想要查询某条公交线路的详细信息，比如途经的站点。"
# seed_queries = [  
#     "911路公交车都经过哪些站？", 
#     "查一下15路",
#     "告诉我虹桥枢纽4路的路线"
# ]

intent_name = "开灯"
intent_description = "用户的核心需求是让灯具 “开启”（如打开房间主灯、台灯、走廊灯等）。"
seed_queries = [
"把灯打开",
"开灯",
"房间灯打开",
"卧室很黑",
"开一下客厅的灯"
]

augmented_queries = generate_similar_queries(intent_name, intent_description, seed_queries, 20)

print("\n--- Generated Queries ---")  
print(json.dumps(augmented_queries, indent=2, ensure_ascii=False))