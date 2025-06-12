import json
import random
import os
import openai
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

# 필요한 함수들 정의
def load_scanrefer_queries():
    scanrefer_queries = []
    with open("./data/scanrefer/ScanRefer_filtered_val.json") as f:
        j_data = json.load(f)
        random.shuffle(j_data)
        for j_query in j_data:
            query = j_query["description"].lower().strip()
            scanrefer_queries.append(query)
    return scanrefer_queries

async def process_query(openai_client, model_name, query):
    # 프롬프트 생성
    prompt = f"""
    Extract the objects, attributes, and relationships from the following query and format the output as three separate lists.
    Use the most basic attributes and relations. For example, "next to" should be categorized as "location", "black" as "color", etc.

    Query: "{query}"
    
    Format:
    Objects: [object_name, ...]
    Attributes: [attribute1, attribute2, ...]
    Relationships: [relation1, ...]
    
    Example:
    Query: "there is a black cat next to a circle ball."
    Objects: [cat, ball]
    Attributes: [color, shape]
    Relationships: [location]
    """

    # LLM 모델 호출
    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # 모델 응답 추출
    result = response.choices[0].message.content.strip()
    return result

async def main():
    # scanrefer 데이터셋 로드
    print("loading scanrefer dataset.")
    all_queries = load_scanrefer_queries()

    if not all_queries:
        print("no queries are loaded. exit.")
        return

    # OpenAI API 키 설정
    openai_client = AsyncOpenAI(
        api_key="",
        base_url="https://generativelanguage.googleapis.com/v1beta/"
    )
    model_name = "gemini-2.0-flash"

    # 모든 쿼리에 대해 비동기적으로 50개씩 처리
    objects = []
    attributes = []
    relationships = []
    for i in tqdm(range(0, len(all_queries), 50), desc="Processing queries"):
        batch_queries = all_queries[i:i+50]
        tasks = [process_query(openai_client, model_name, query) for query in batch_queries]
        results = await asyncio.gather(*tasks)

        # 결과를 파싱하여 각 배열에 추가
        for result in results:
            lines = result.split('\n')
            for line in lines:
                if line.startswith("Objects:"):
                    objects.extend(line[len("Objects:"):].strip().strip('[]').split(', '))
                elif line.startswith("Attributes:"):
                    attributes.extend(line[len("Attributes:"):].strip().strip('[]').split(', '))
                elif line.startswith("Relationships:"):
                    relationships.extend(line[len("Relationships:"):].strip().strip('[]').split(', '))

        # 각 배치 처리 후 지연 시간 추가
        await asyncio.sleep(1)  # 1초 지연

    # 결과를 텍스트 파일로 저장
    with open("results.txt", "w") as f:
        f.write("Objects:\n")
        f.write(", ".join(objects) + "\n")
        f.write("Attributes:\n")
        f.write(", ".join(attributes) + "\n")
        f.write("Relationships:\n")
        f.write(", ".join(relationships) + "\n")

    print("Results saved to results.txt")

if __name__ == "__main__":
    asyncio.run(main())