import json  
import os  
from openai import OpenAI  
# LangChain 相关库  
# 需要安装: pip install langchain langchain-openai faiss-cpu tiktoken  
from langchain_community.vectorstores import FAISS  
from langchain_openai import OpenAIEmbeddings  
from langchain.schema import Document

# --- 0. 准备工作 ---  
# 这是我们构建的意图识别知识库 ，第一次建立需要注入数据到磁盘，后续可无需重复注入数据
knowledge_base = [  
    {"query": "来个睡前故事吧", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "有啥睡前故事推荐吗", "intent": "用户想听睡前故事", "slots": {"story_type": "推荐"}},
    {"query": "给我讲个睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "能讲个睡前故事吗", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "睡前有没有故事", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "来个睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "有啥适合睡前听的", "intent": "用户想听睡前故事", "slots": {"story_type": "助眠"}},
    {"query": "讲个睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "有啥助眠的睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type": "助眠"}},
    {"query": "睡前故事能听啥", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "睡前故事有啥推荐", "intent": "用户想听睡前故事", "slots": {"story_type": "推荐"}},
    {"query": "能讲个睡前故事给陪一下", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "睡前故事讲讲", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "睡前故事有些啥", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "睡前故事能听啥内容", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "有啥温馨的睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type":  "温馨"}},
    {"query": "睡前故事讲个啥", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "能给讲个睡前故事吧", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}},
    {"query": "有啥适合睡前听的故事", "intent": "用户想听睡前故事", "slots": {"story_type": "助眠"}},
    {"query": "我能听啥睡前故事", "intent": "用户想听睡前故事", "slots": {"story_type": "任意"}}, 
    {"query": "做眼保健操了吗？", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "该做眼操了，对吧？", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "感觉眼睛有点沉闷，该做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "我的眼睛感觉有点干涩，该做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "现在做眼操行不行？", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点累，该休息一下了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "做眼操的时候了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点涩，该做做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "我的眼睛有点累，快点做眼操吧", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "我的眼睛有点沉，该做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "我的眼睛有点干，需要做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "该做眼操了，别忘了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点模糊，该做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "做眼操的时间到了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点酸，做眼操一下", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "该做眼操了，你记住了吗？", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点困，做眼操吧", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "我的眼睛有点涩，该做眼操了", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "想做眼操了，可以做吗？", "intent": "用户应做眼保健操", "slots": {"trigger_type": "用户主动询问"}},
    {"query": "我的眼睛有点沉，去做眼操吧", "intent": "用户应做眼保健操", "slots": {"trigger_type": "状态触发提醒"}},
    {"query": "客人到了，快来招呼", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "门口有人，快去迎接", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "来客人了，赶紧迎一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "有客人，去接一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "客人来了，赶紧迎过去", "intent": "迎宾", "slots": {"guest_location": "附近"}},
    {"query": "快去把客人迎进来", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "欢迎客人，门口有人吗", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "来客人了，快去迎一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "门口来了人，赶紧去迎接", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "客人到了，快去接一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "有客人，赶紧去迎接", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "门口有人，快去招呼", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "客人来了，赶紧去接一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "客人到了，去迎接一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "客人来了，赶紧迎迎", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "门口有人，快去迎接", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "有客人，赶紧去接一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "有客人，赶紧去迎一下", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "客人来了，赶紧去迎迎", "intent": "迎宾", "slots": {"guest_location": "未知（默认）"}},
    {"query": "门口有人，快去招呼", "intent": "迎宾", "slots": {"guest_location": "门口"}},
    {"query": "别让那位客人进门，不要迎接他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝迎接"}},
    {"query": "让这位客人在外面等，别让他进来", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "要求等待+禁止进门"}},
    {"query": "别开门让客人进来，不欢迎他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止开门+禁止进门"}},
    {"query": "我不认识那个客人，别让他进来", "intent": "拒迎宾客", "slots": {"guest_specificity": "陌生客人", "rejection_action": "禁止进门"}},
    {"query": "这位客人别想进来，不欢迎他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+明确不欢迎"}},
    {"query": "别让他进来，我不要见他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝见面"}},
    {"query": "别迎接这位客人，我不认识他", "intent": "拒迎宾客", "slots": {"guest_specificity": "陌生客人）", "rejection_action": "禁止进门+拒绝迎接"}},
    {"query": "不欢迎他，别让他进来", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "明确不欢迎+禁止进门"}},
    {"query": "别开门让他进来，我不见他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止开门+拒绝见面"}},
    {"query": "别让他进来，我不要接待他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝接待"}},
    {"query": "别让客人进来，不欢迎他", "intent": "拒迎宾客", "slots": {"guest_specificity": "未明确（默认）", "rejection_action": "禁止进门+明确不欢迎"}},
    {"query": "别开门让他进来，我不要见他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止开门+拒绝见面"}},
    {"query": "不让他进来，别迎接他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝迎接"}},
    {"query": "客人别进来，我不要见他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝见面"}},
    {"query": "别开门让他进来，不要接待他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止开门+拒绝接待"}},
    {"query": "别让他进来，我不欢迎他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+明确不欢迎"}},
    {"query": "别迎接他，别让他进来", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "拒绝迎接+禁止进门"}},
    {"query": "不欢迎他，让他在外面等", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "明确不欢迎+要求等待"}},
    {"query": "别让他进来，我不要接待他", "intent": "拒迎宾客", "slots": {"guest_specificity": "特定客人", "rejection_action": "禁止进门+拒绝接待"}},
    {"query": "把灯打开", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "开灯了", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "把房间的灯打开", "intent": "开灯", "slots": {"light_location": "房间", "request_type": "直接指令"}},
    {"query": "卧室灯暗", "intent": "开灯", "slots": {"light_location": "卧室", "request_type": "状态暗示"}},
    {"query": "客厅灯关着", "intent": "开灯", "slots": {"light_location": "客厅", "request_type": "状态暗示"}},
    {"query": "我要开灯", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "房间太暗了", "intent": "开灯", "slots": {"light_location": "房间", "request_type": "状态暗示"}},
    {"query": "能不能开灯", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "询问许可"}},
    {"query": "打开灯光", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "走廊灯没亮", "intent": "开灯", "slots": {"light_location": "走廊", "request_type": "状态暗示"}},
    {"query": "把灯给我开开", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "现在很黑", "intent": "开灯", "slots": {"light_location": "当前环境", "request_type": "状态暗示"}},
    {"query": "我能开灯吗", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "询问许可"}},
    {"query": "房间的灯关着", "intent": "开灯", "slots": {"light_location": "房间", "request_type": "状态暗示"}},
    {"query": "能把灯打开吗", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "询问许可"}},
    {"query": "台灯关着，房间很暗", "intent": "开灯", "slots": {"light_location": "房间", "request_type": "状态暗示"}},
    {"query": "我要亮灯", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "直接指令"}},
    {"query": "客厅灯关了", "intent": "开灯", "slots": {"light_location": "客厅", "request_type": "状态暗示"}},
    {"query": "把走廊灯打开", "intent": "开灯", "slots": {"light_location": "走廊", "request_type": "直接指令"}},
    {"query": "能把灯打开了吗", "intent": "开灯", "slots": {"light_location": "任意", "request_type": "询问许可"}},
]

# --- 配置 API ---  

# 实例化 client 和 embeddings    
api_key = "ollama"
base_url = "http://localhost:11434"  # 移除/v1，作为基础URL

# 客户端使用带/v1的URL（对话接口）
client = OpenAI(api_key=api_key, base_url=f"{base_url}/v1")  

# 使用Ollama的embedding模型
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url=base_url,  # 使用不带/v1的基础URL(向量嵌入接口)
)

# --- 1. 使用 LangChain 构建向量知识库 ---  
print("Step 1: Building vector store with LangChain...")  
VECTOR_STORE_PATH = "/home/xumingcong/Vector-data"   #向量数据库存储地址
try:
    # 第一步：尝试从磁盘加载已有的向量库
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Found existing vector store at {VECTOR_STORE_PATH}, loading...")
        # 从磁盘加载：需指定路径和嵌入模型（嵌入模型必须与保存时一致！）
        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 允许反序列化（FAISS存储的元数据需此配置）
        )
        print("Vector store loaded from disk successfully.")
    else:
        # 第二步：磁盘无向量库，从 knowledge_base 构建并保存到磁盘
        print(f"No existing vector store found. Building new one...")
        # 1. 将原始知识库转换为 LangChain Document 格式（与原逻辑一致）
        documents = [  
            Document(  
                page_content=item['query'],  
                metadata={'intent': item['intent'], 'slots': json.dumps(item['slots'])}  
            ) for item in knowledge_base  
        ]
        # 2. 构建 FAISS 向量库（与原逻辑一致）
        vector_store = FAISS.from_documents(documents, embeddings)
        # 3. 关键：将构建好的向量库保存到磁盘
        vector_store.save_local(folder_path=VECTOR_STORE_PATH)
        print(f"Vector store built and saved to {VECTOR_STORE_PATH} successfully.")  
except Exception as e:  
    print(f"Error building vector store: {e}")  
    vector_store = None

def retrieve_examples_langchain(user_query, k=3):
    """  
    使用 LangChain 的向量存储检索最相似的K个示例。  
    """
    print(f"nStep 2: Retrieving examples for query: '{user_query}' with LangChain")  
    if not vector_store:  
        print("Vector store is not available.")  
        return []  
          
    # FAISS.similarity_search会返回 Document 对象列表  
    retrieved_docs = vector_store.similarity_search(user_query, k=k)  
      
    # 将 Document 对象转换回我们原来的字典格式，以便下游函数使用  
    examples = [  
        {  
            "query": doc.page_content,  
            "intent": doc.metadata['intent'],  
            "slots": json.loads(doc.metadata['slots'])  
        } for doc in retrieved_docs  
    ]  
    print(f"Retrieved {len(examples)} examples.")  
    return examples

def build_prompt_with_rag(user_query, examples):
    """  
    构建带有检索到的示例的动态提示词。  
    """
    print("\nStep 3: Building dynamic prompt with retrieved examples...")  
    examples_str = "\n".join([f"// 示例\n用户输入: {ex['query']}\n输出: {json.dumps({'intent': ex['intent'], 'slots': ex['slots']}, ensure_ascii=False)}"for ex in examples])

    prompt = f"""  
    你是一个任务型对话机器人的NLU（自然语言理解）引擎。  
    你的任务是根据用户最新的提问，识别出用户的意图（intent）并抽取出相应的槽位（slots）。

    请严格参考下面提供的示例，理解如何进行意图识别和槽位抽取。

    {examples_str}

    ---  
    现在，请处理以下用户的最新提问。  
    请严格按照JSON格式输出，不要包含任何其他解释。

    用户输入: {user_query}  
    输出:  
    """
    print("Prompt built.")  
    return prompt

def recognize_intent_with_rag(user_query):
    """  
    执行完整的RAG意图识别流程。  
    """
    # 1. 检索 (使用 LangChain 版本)  
    examples = retrieve_examples_langchain(user_query)  
      
    # 2. 构建提示词  
    prompt = build_prompt_with_rag(user_query, examples)  
    print("\n--- Final Prompt to LLM ---")  
    print(prompt)  
    print("---------------------------")

    # 3. 调用LLM  
    print("\nStep 4: Calling LLM for final recognition...")  
    try:  
        response = client.chat.completions.create(  
            model="qwen3",  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0, # 对于分类和提取任务，使用低温  
            response_format={"type": "json_object"},  
        )  
        result = response.choices[0].message.content  
        print("LLM call successful.")  
        return json.loads(result)  
    except Exception as e:  
        print(f"An error occurred during LLM call: {e}")  
        return {"error": str(e)}

# --- 最终测试 ---  
if vector_store:  
    test_query_1 = "卧室很黑"
    result_1 = recognize_intent_with_rag(test_query_1)  
    print(f"\n--- Result for '{test_query_1}' ---")  
    print(json.dumps(result_1, indent=2, ensure_ascii=False))

    test_query_2 = "门外这个人我不认识"
    result_2 = recognize_intent_with_rag(test_query_2)  
    print(f"\n--- Result for '{test_query_2}' ---")  
    print(json.dumps(result_2, indent=2, ensure_ascii=False))