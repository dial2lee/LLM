from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

model = OllamaFunctions(model="qwen2:7b", format='json')

def search_web(query, k=5, max_retry=3):
#     import os
#     from langchain_community.retrievers import TavilySearchAPIRetriever

#     os.environ["TAVILY_API_KEY"]=tavily_api_key

#     retriever = TavilySearchAPIRetriever(k=5)

#     documents = retriever.invoke(query)

#     content = '\n\n'.join([doc.page_content for doc in documents])

#     prompt = f"""请将下面这段内容（<<<content>>><<</content>>>包裹的部分）进行总结：
# <<<content>>>
# {content}
# <<</content>>>
# """
#     print('prompt:')
#     print(prompt)

#     return model.invoke(prompt).content
    return ""

def get_current_weather(city):
    print('get weather for:' + city)
    # import requests
    # from datetime import datetime
    
    # url = f'https://restapi.amap.com/v3/config/district?keywords={city}&key={amap_api_key}'
    # resp = requests.get(url)
    # # print('行政区划：')
    # # print(resp.json())
    
    # adcode = resp.json()['districts'][0]['adcode']
    # # adcode = '110000'

    # url = f'https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={amap_api_key}&extensions=base'
    # resp = requests.get(url)
    # """样例数据
    # {'province': '北京',
    #  'city': '北京市',
    #  'adcode': '110000',
    #  'weather': '晴',
    #  'temperature': '26',
    #  'winddirection': '西北',
    #  'windpower': '4',
    #  'humidity': '20',
    #  'reporttime': '2024-05-26 13:38:38',
    #  'temperature_float': '26.0',
    #  'humidity_float': '20.0'}
    # """
    # # print('天气：')
    # # print(resp.json())
    # weather_json = resp.json()['lives'][0]
    # return f"{weather_json['city']}{datetime.strptime(weather_json['reporttime'], '%Y-%m-%d %H:%M:%S').strftime('%m月%d日')}{weather_json['weather']}，气温{weather_json['temperature']}摄氏度，{weather_json['winddirection']}风{weather_json['windpower']}级"
    return "北京市05月26日晴，气温25摄氏度，西风≤3级"
     
fn_map = {
    'get_current_weather': get_current_weather,
    'search_web': search_web
}

llm_with_tool = model.bind(
    functions=[
        {
            "name": "get_current_weather",
            "description": "根据城市名获取天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如北京",
                    }
                },
                "required": ["city"],
            },
        },
        {
            "name": "search_web",
            "description": "搜索互联网",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜素的内容"
                    }
                },
                "required": ["query"]
            }
        },
    ],
    # function_call={"name": "get_current_weather"}
)

from langchain_core.messages import HumanMessage, ToolMessage
import json

query = "北京明天的天气怎么样"
messages = [HumanMessage(query)]

# {'function_call': {'name': 'get_current_weather', 'arguments': '{"city": "\u5317\u4eac"}'}}
ai_msg = llm_with_tool.invoke(messages)

messages.append(ai_msg)

# {'city': '北京'}
kwargs = json.loads(ai_msg.additional_kwargs['function_call']['arguments'])

tool_output = fn_map[ai_msg.additional_kwargs['function_call']['name']](**kwargs)

print("tool output:" + tool_output)

messages.append(ToolMessage(tool_output, tool_call_id=ai_msg.id))

ollamaModel = Ollama(model="qwen2:7b")

final_output = ollamaModel.invoke(messages)

print("final output:" + final_output)