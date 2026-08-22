import asyncio
import csv
import json
import re
import os
import time

import concurrent.futures
from http.client import responses

import function
import random
import numpy as np
# import visualization
from deap import creator, tools, base, algorithms
from ollama import Client,AsyncClient,ResponseError
from typing import List, Dict, Any, Awaitable, Tuple, Optional
import warnings
# from route_planning.NSGA2Bus import toolbox
# from function import readText
from function import *
from openai import OpenAI, AsyncOpenAI

from data_loader import load_data
from utils import create_compatible_env

# T12: LLM 调用追踪器
from llm_tracker import LLMTracker, tracked_chat_create


# ================= 算子性能记录器（每次迭代记录 cx/mt 精英分与平均分） =================
class _OperatorPerformanceRecorder:
    """
    模块级单例：记录 cx/mt 进化分支每次迭代的算子池精英分与平均分。

    设计目标：
    - 每次运行 agent 时，reset() 清空旧记录；
    - 在 cx/mt 进化主循环的每代末尾调用 record() 采集指标；
    - 全部分支运行结束后，save() 统一写入 temp/operator_performance_iter.json；
    - 即使某一分支被注释未运行，输出 JSON 中仍保留该分支的空列表字段。
    """

    _BRANCHES = ("cx", "mt")
    _DEFAULT_SAVE_PATH = "./temp/operator_performance_iter.json"

    def __init__(self):
        self._data = {b: [] for b in self._BRANCHES}

    def reset(self):
        """清空所有分支的记录（每次运行 agent 时调用）。"""
        self._data = {b: [] for b in self._BRANCHES}

    def record(self, branch, iteration, elite_scores, avg_scores):
        """
        记录某分支单次迭代的表现。

        参数:
            branch (str): "cx" 或 "mt"
            iteration (int): 迭代号（从 1 开始，与 CSV 的 Generation 列一致）
            elite_scores (tuple): (elite_overall, elite_visual, elite_demand)
            avg_scores (tuple): (avg_overall, avg_visual, avg_demand)
        """
        if branch not in self._BRANCHES:
            return
        eo, ev, ed = elite_scores
        ao, av, ad = avg_scores
        self._data[branch].append({
            "iteration": int(iteration),
            "elite_overall": float(eo),
            "elite_visual": float(ev),
            "elite_demand": float(ed),
            "avg_overall": float(ao),
            "avg_visual": float(av),
            "avg_demand": float(ad),
        })

    def save(self, path=None):
        """将所有分支记录写入 JSON（覆盖旧文件）；未运行的分支以空列表占位。"""
        path = path or self._DEFAULT_SAVE_PATH
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            print(f"[性能记录] 已保存到 {path} (cx: {len(self._data['cx'])} 条, mt: {len(self._data['mt'])} 条)")
        except Exception as e:
            print(f"[性能记录] 保存失败: {e}")


# 模块级单例
_operator_performance_recorder = _OperatorPerformanceRecorder()


# API Key 改为环境变量读取（T12 Step 7）
ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
if not ALIYUN_API_KEY:
    print("[警告] 未设置 ALIYUN_API_KEY 环境变量；LEHHA 进化将无法调用 API")


def extract_true_response(response_text: str) -> str:
    """
    从 Ollama 响应中移除 <think>...</think> 块，只保留真正的回复内容。
    """
    if not response_text:
        return ""
    # flags=re.DOTALL : 让 . 号也能匹配换行符，确保多行思考也能被选中
    pattern = r'<think>.*?</think>'
    # 将匹配到的 <think>...</think> 块替换为空字符串
    cleaned_text = re.sub(pattern, '', response_text, flags=re.DOTALL)
    # 去除首尾可能的空白字符 (换行符、空格)
    return cleaned_text.strip()


def extract_code_block(result: str) -> Optional[str]:
    """
    从LLM响应中提取第一个 ```python ... ``` 代码块中的内容。
    即使前后有说明文字或多个代码块，也能准确提取。
    """
    if not result:
        return None

    # 1. 优先匹配 ```python 块
    pattern_python = r"```python\s*([\s\S]*?)\s*```"
    match_python = re.search(pattern_python, result, re.IGNORECASE)

    if match_python:
        # 提取捕获组 1 (即代码内容)
        return match_python.group(1).strip()

    # 2. 如果没有 ```python，则匹配 ``` 块
    pattern_generic = r"```\s*([\s\S]*?)\s*```"
    match_generic = re.search(pattern_generic, result)

    if match_generic:
        return match_generic.group(1).strip()

    # 3. 如果没有代码块标记，返回原始文本
    return result.strip()

def call_llm(prompt):
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    model_name = 'qwen-plus'  # 改为 qwen-plus

    prompt_template = """

      """

    prompt = prompt_template.format(

    )
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = tracked_chat_create(client, phase="other",
        model=model_name,
        messages=messages
    )

    try:
        json_output = response.choices[0].message.content.strip()
        output_data = json.loads(json_output)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print('aaa')

def get_summary(evolution_history):
    evolution_history = json.dumps(evolution_history, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
    # 1. 初始化 OpenAI 客户端
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    model_name = 'qwen-plus'

    # 2. 定义提示词模板
    system_prompt = readText("./prompts/system_prompt.txt")
    user_prompt_template = readText("./prompts/summary_prompt.txt")

    # 3. 填充提示词并构造消息格式
    user_prompt = user_prompt_template.format(
        problem_description=readText("./description/description.txt"),
        evolution_history = evolution_history
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 4. 调用 OpenAI API (qwen-plus)
    response = tracked_chat_create(client, phase="other",
        model=model_name,
        messages=messages
    )

    result = response.choices[0].message.content.strip()

    print("get_summary返回结果：",result)

    return result

def llm_tuning_agent(pop_info,n_gen,gen = 0,message = None,info=False,retry = 5):
    # # 手动定义调试用的NSGA-II参数（可根据需求修改数值）
    # debug_params = {
    #     "maxGen": 5,  # 初始最大迭代次数（调试用示例值）
    #     "pop_size": 20,  # 种群规模（调试用示例值）
    #     "cxProb": 0.85,  # 交叉概率（调试用示例值）
    #     "mutateProb": 0.03  # 变异概率（调试用示例值）
    # }
    # return debug_params

    if info:#非初次参数生成，有额外信息
        info_json_str = json.dumps(pop_info, ensure_ascii=False, indent=2)
        # stage_1_str = json.dumps(stage_1, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
        # stage_2_str = json.dumps(stage_2, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
        # stage_3_str = json.dumps(stage_3, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
        # 1. 初始化 Ollama 客户端
        ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
        BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        MODEL_NAME = "qwen-plus"
        sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

        # 2. 定义提示词模板
        system_prompt = readText("./prompts/system_prompt.txt")
        user_prompt_template = readText("./prompts/tuning_prompt_with_info.txt")

        # 3. 填充提示词并构造 Ollama 消息格式
        user_prompt = user_prompt_template.format(
            problem_desc = readText("./description/description.txt"),
            current_gen = gen,
            maxGen = n_gen,
            info=info_json_str,
        )

        # prompt = prompt_template.replace("{solver}", solver.strip())
        # prompt = prompt.replace("{keyword}", keyword.strip())
        # prompt = prompt.replace("{context}", context_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if message is not None:
            messages.append({"role": "user", "content": message})
        # 4. 调用 Ollama 模型（强制 JSON 格式输出）
        response = tracked_chat_create(sync_client, phase="other",
            model=MODEL_NAME,
            messages=messages
        )

        result = response.choices[0].message.content.strip()

        print("tuning_with_info返回结果：", result)

        pattern =  r"\{[\s\S]*?\}"

        matches = re.findall(pattern, result)
        if matches:
            result = matches[-1]
            try:
                result = json.loads(result)
                print("提取结果：")
                print(result)
                try:
                    if result["cxProb"] and result["mutateProb"]:
                        return result

                except Exception as e:

                    if retry > 0:
                        print("尝试修复tuning操作，修复次数：", 5 - retry)
                        message = (
                            "Your previous output failed JSON format parsing. The error message is: "
                            f"'{str(e)}'. "
                            "***CRITICAL INSTRUCTION: You MUST analyze the provided 'evolution_history' and 'summary_detail' to ADJUST the parameters based on the problem's performance. DO NOT output standard/default values.*** "
                            "Please regenerate a strictly compliant JSON format output. "
                            "The content must be in the format {'maxGen':int,'pop_size':int,'cxProb':float,'mutateProb':float}."
                        )
                        # return llm_tuning_agent(evolution_history_str, summary_detail, message=message, info=True,
                        #                         retry=retry - 1)
                        return evolution_history[-1]["method"]
                    else:
                        print("修复失败")
                        return {}
            except json.JSONDecodeError as e:
                if retry > 0:
                    print("尝试修复tuning操作，修复次数：", 5 - retry)
                    message = (
                        "Your previous output failed JSON format parsing. The error message is: "
                        f"'{str(e)}'. "
                        "***CRITICAL INSTRUCTION: You MUST analyze the provided 'evolution_history' and 'summary_detail' to ADJUST the parameters based on the problem's performance. DO NOT output standard/default values.*** "
                        "Please regenerate a strictly compliant JSON format output. "
                        "The content must be in the format {'maxGen':int,'pop_size':int,'cxProb':float,'mutateProb':float}."
                    )
                    # return llm_tuning_agent(evolution_history_str, summary_detail,message=message,info = True, retry=retry - 1)
                    return evolution_history[-1]["method"]
                    # return evolution_history_str[-1].get("method", {})
                print("修复失败")
                return {}
        else:
            print("未找到字典内容")
    else:#初次生成参数
        # evolution_history_str = json.dumps(stage_1, stage_2, stage_3, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
        # 1. 初始化 Ollama 客户端
        ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
        BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        MODEL_NAME = "qwen-plus"
        sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

        # 2. 定义提示词模板
        system_prompt = readText("./prompts/system_prompt.txt")
        user_prompt_template = readText("./prompts/tuning_prompt.txt")

        # 3. 填充提示词并构造 Ollama 消息格式
        user_prompt = user_prompt_template.format(
            problem_desc = readText("./description/description.txt"),
        )

        # prompt = prompt_template.replace("{solver}", solver.strip())
        # prompt = prompt.replace("{keyword}", keyword.strip())
        # prompt = prompt.replace("{context}", context_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if message is not None:
            messages.append({"role": "user", "content": message})
        # 4. 调用 Ollama 模型（强制 JSON 格式输出）
        response = tracked_chat_create(sync_client, phase="other",
            model=MODEL_NAME,
            messages=messages
        )

        result = response.choices[0].message.content.strip()

        print("tuning返回结果：", result)
        pattern = r"\{[\s\S]*?\}"

        matches = re.findall(pattern, result)
        if matches:
            result = matches[-1]
            try:
                result = json.loads(result)
                print("提取结果：")
                print(result)
                return result
            except json.JSONDecodeError as e:
                if retry > 0:
                    print("尝试修复tuning操作，修复次数：", 5 - retry)
                    message = (
                        "Your previous output failed JSON format parsing. The error message is: "
                        f"'{str(e)}'. "
                        "***CRITICAL INSTRUCTION: You MUST analyze the provided 'evolution_history' and 'summary_detail' to ADJUST the parameters based on the problem's performance. DO NOT output standard/default values.*** "
                        "Please regenerate a strictly compliant JSON format output. "
                        "The content must be in the format {'maxGen':int,'pop_size':int,'cxProb':float,'mutateProb':float}."
                    )
                    return {"cxProb": 0.6, "mutateProb": 0.4}
                else:
                    print("修复失败")
                    return {}
        else:
            print("未找到字典内容")


async def generate_single_operator(
        client: Any,  # 实际类型应为 AsyncClient
        env: Any,
        base_prompt: str,
        system_prompt: str,
        index: int,
        evaluate_pop: list,
        toolbox: Any,
        op_type: str,
        current_retry: int = 0,
        max_retries: int = 3,
        last_error: str = None,
        last_code: str = None,
        model_name:str = "qwen-plus",
) -> str:
    """
    生成一个通用算子（交叉或变异），如果评估报错或超时，则递归调用自身让 LLM 修复。
    仅对生成的算子进行评估时设置超时。
    """
    EVALUATION_TIMEOUT_SECONDS = 60

    # --- 1. 初始化和 Prompt 构建 (保持不变) ---

    code_str = ""  # 预定义 code_str，用于在 Try-Run 失败时传入

    if op_type == "cx":
        op_name = "crossover"
        func_req_desc = "function must accept (ind1, ind2, env) and return two valid children in a tuple or list."
    elif op_type == "mt":
        op_name = "mutation"
        func_req_desc = "function must accept (input_ind, env) and return a modified route (list of nodes)."

    if current_retry == 0:
        print(f"开始生成 {op_name} {index}...")
        final_user_prompt = base_prompt
    else:
        print(f"{op_name} {index} 正在进行第 {current_retry}/{max_retries} 次自我修复...原因：{last_error}")
        repair_instruction = (
            f"\n\n[ERROR REPORT]\n"
            f"Your previous generated code failed to execute or timed out during evaluation.\n"
            f"Previous Code:\n```python\n{last_code}\n```\n"
            f"Error Message:\n{last_error}\n\n"
            f"[INSTRUCTION]\n"
            f"Please analyze the error and FIX the code. "
            f"Requirement: The {func_req_desc} "
            f"Output ONLY the fixed Python code block for the {op_name}."
        )
        final_user_prompt = f"{base_prompt}{repair_instruction}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_prompt}
    ]

    try:
        # --- 2. LLM 调用 (无超时限制) ---
        try:
            response = await tracked_chat_create(client, phase="init",
                model=model_name,
                messages=messages,
                temperature=0.7 if current_retry > 0 else 0.9,
                timeout=300
            )
            result = response.choices[0].message.content

            code_str = extract_code_block(extract_true_response(result))
            # print(code_str)
            if not code_str:
                raise ValueError("No code block found in LLM response.")

        except Exception as e:
            # 捕获 LLM 调用的网络错误、连接中断等
            raise Exception(f"LLM Chat Error: {str(e)}")

        # --- 3. 立即评估 (Try-Run) ---
        validation_error = None
        op_func = None  # 预定义 op_func

        try:
            # A. 编译
            op_func = compile_and_load_code_string(code_str)
            if not op_func:
                raise ValueError("Compilation failed (syntax error or no function found).")

            # B. 运行测试 (使用 asyncio.to_thread 和 wait_for 实现超时)
            def _run_evaluation_tests():
                # ** 变异算子 (Mutation Operator: 'mt') **
                if op_type == 'mt':
                    # 只测前 4 个个体，节省时间
                    test_parents = evaluate_pop[:4]
                    if not test_parents: return

                    for parent in test_parents:
                        p_copy = toolbox.clone(parent)
                        p_copy = convert_individual_nodes_to_int(p_copy)

                        # 直接调用变异函数 op_func(individual, env)
                        children = op_func(p_copy, env)

                        # 严格检查返回值格式 (DEAP 要求返回元组)
                        if not isinstance(children, tuple):
                            raise ValueError(f"Mutation operator must return a tuple, got {type(children)}.")
                        if len(children) != 1:
                            raise ValueError(f"Mutation operator tuple length must be 1, got {len(children)}.")

                        child = children[0]
                        if not child or len(child) == 0:
                            raise ValueError("Mutation operator returned empty child individual.")
                        if not isinstance(child, list):  # 确保是个体列表
                            raise ValueError(f"Mutation child must be a list, got {type(child)}.")

                # ** 交叉算子 (Crossover Operator: 'cx') **
                elif op_type == 'cx':
                    pairs = list(itertools.combinations(evaluate_pop, 2))
                    test_pairs = pairs[:2] if len(pairs) > 2 else pairs
                    if not test_pairs: return

                    for parent1, parent2 in test_pairs:
                        p1_copy, p2_copy = toolbox.clone(parent1), toolbox.clone(parent2)
                        p1_copy = convert_individual_nodes_to_int(p1_copy)
                        p2_copy = convert_individual_nodes_to_int(p2_copy)

                        # 直接调用交叉函数 op_func(ind1, ind2, env)
                        children = op_func(p1_copy, p2_copy, env)

                        # 严格检查返回值格式 (DEAP 要求返回元组)
                        if not isinstance(children, tuple):
                            raise ValueError(f"Crossover operator must return a tuple, got {type(children)}.")
                        if len(children) != 2:
                            raise ValueError(f"Crossover operator tuple length must be 2, got {len(children)}.")

                        child1, child2 = children
                        if not child1 or len(child1) == 0 or not child2 or len(child2) == 0:
                            raise ValueError("Crossover operator returned empty children.")

                else:
                    raise ValueError(f"Unknown operator type: {op_type}")

                return True

            # 实际执行：在线程中运行评估逻辑，并设置超时
            await asyncio.wait_for(
                asyncio.to_thread(_run_evaluation_tests),
                timeout=EVALUATION_TIMEOUT_SECONDS
            )

            # C. 如果运行到这里没有报错，说明代码是“可运行的”
            print(f"✅ {op_name} {index} 已生成并通过 Try-Run。")
            return code_str

        except asyncio.TimeoutError:
            # 捕获评估超时
            validation_error = f"Evaluation timed out after {EVALUATION_TIMEOUT_SECONDS} seconds."

        except Exception as e:
            # 捕获编译错误、运行时错误、验证失败等
            import traceback
            tb_lines = traceback.format_exc().splitlines()
            # 提取最后几行 Traceback 信息，帮助 LLM 定位错误
            relevant_tb = "\n".join(tb_lines[-3:]) if len(tb_lines) >= 3 else str(e)
            validation_error = (
                f"{type(e).__name__}: {str(e)}\n"
                f"Traceback Snippet:\n{relevant_tb}"
            )

        # 4. 错误处理与递归
        if validation_error:
            if current_retry < max_retries:
                # 递归调用进行修复
                return await generate_single_operator(
                    client, env, base_prompt, system_prompt, index,
                    evaluate_pop, toolbox, op_type,
                    current_retry + 1, max_retries,
                    last_error=validation_error,
                    last_code=code_str,  # 传入本次生成的代码
                    model_name=model_name
                )
            else:
                # 超过重试次数，放弃治疗
                print(f"{op_name} {index} 超过最大重试次数，放弃。最后错误: {validation_error}")
                return f"# Generation Failed after {max_retries} retries.\n# Error: {validation_error}\n{code_str}"

    except Exception as e:
        # 处理 LLM 调用失败（网络、连接错误）
        error_message = str(e)
        if current_retry < max_retries:
            # 进入修复模式
            return await generate_single_operator(
                client, env, base_prompt, system_prompt, index,
                evaluate_pop, toolbox, op_type,
                current_retry + 1, max_retries,
                last_error=error_message,
                # 如果是 LLM 错误，则没有 code_str 可传入，使用提示信息
                last_code="LLM Chat Error" if current_retry == 0 else last_code,
                model_name=model_name
            )
        else:
            print(f"{op_name} {index} 超过最大重试次数，放弃。系统错误: {error_message}")
            return f"# System Error during generation of {op_name}: {error_message}"


async def generate_single_reflection(client: AsyncOpenAI, better: Dict, worse: Dict, model: str = "qwen-plus", direction: str = "Overall") -> str:
    """
    辅助函数：处理单个成对的反射请求

    Args:
        direction: 反思方向 (Overall/Visual/Demand)，将添加到生成内容的前缀
    """
    prompt_template = readText("./prompts/st_reflection_prompt.txt")
    prompt = prompt_template.format(
        problem_desc = readText("./description/description.txt"),
        worse_code=worse['code'],
        better_code=better['code']
    )

    try:
        response = await tracked_chat_create(client, phase="short_reflection",
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            timeout=300
        )
        llm_response_text = response.choices[0].message.content.strip()

        if not llm_response_text:
            return ""
        response_text = extract_true_response(llm_response_text)

        # 添加方向前缀
        direction_definitions = {
            "Overall": "Optimize for Pareto Dominance (better trade-off) and Path Diversity (novel routes).",
            "Visual": "Optimize specifically for Objective 1: Path Compactness, Aesthetics, and geometric efficiency.",
            "Demand": "Optimize specifically for Objective 2: Demand Satisfaction, Load Balancing, and Capacity constraints."
        }
        target_goal = direction_definitions.get(direction, "Optimize the operator.")
        direction_prefix = f"{target_goal}"
        enhanced_response = direction_prefix + response_text

        print(enhanced_response[:100])
        return enhanced_response
    except Exception as e:
        print(f"Reflection generation failed: {e}")
        return ""


async def short_term_reflection(
        client: AsyncOpenAI,
        population: List[Dict[str, Any]],
        model_name: str = "qwen-plus",
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    对种群进行短时反思 (Short-term Reflection)。
    两两对比，分析三个分数的差异比例，选择差异最大的方向生成反思。

    :return: (hints_list, worse_code_list, better_code_list, direction_list)
    """
    tasks = []
    worse_code_lst = []
    better_code_lst = []
    direction_lst = []  # 新增：记录反思方向

    # 确保种群数量是偶数，如果是奇数，丢弃最后一个或单独处理（这里选择只处理成对的）
    limit = len(population) if len(population) % 2 == 0 else len(population) - 1

    print(f"开始生成短时反思 Hints，共 {limit // 2} 组...")

    def safe_diff(val_a, val_b):
        """安全计算相对差异，处理负数和零值"""
        # 惩罚分（-100）不参与差异计算
        if val_a < -50 or val_b < -50:
            return 0.0
        min_val = min(val_a, val_b)
        max_val = max(val_a, val_b)
        # 都接近0或为负
        if max_val < 0.001:
            return 0.0
        # 使用 abs(max_val) + 1.0 避免除零
        return (max_val - min_val) / (abs(max_val) + 1.0)

    for i in range(0, limit, 2):
        ind_a = population[i]
        ind_b = population[i + 1]

        score_a = ind_a.get('score')
        score_b = ind_b.get('score')

        # 安全检查：确保分数是有效的元组/列表
        if not isinstance(score_a, (tuple, list)) or not isinstance(score_b, (tuple, list)):
            # 如果分数无效，回退到综合分（如果有的话）
            fallback_score_a = score_a[0] if isinstance(score_a, (tuple, list)) else score_a
            fallback_score_b = score_b[0] if isinstance(score_b, (tuple, list)) else score_b
            if fallback_score_a >= fallback_score_b:
                better_ind, worse_ind = ind_a, ind_b
            else:
                better_ind, worse_ind = ind_b, ind_a
            direction = "Overall"
            print(f"  警告: 对组 {i//2 + 1} 的分数格式无效，使用默认方向 Overall")
        else:
            # 计算三个维度的相对差异比例
            diff_total = safe_diff(score_a[0], score_b[0])
            diff_visual = safe_diff(score_a[1], score_b[1])
            diff_demand = safe_diff(score_a[2], score_b[2])

            # 找出差异最大的方向
            diffs = {
                'Overall': diff_total,
                'Visual': diff_visual,
                'Demand': diff_demand
            }
            direction = max(diffs, key=diffs.get)

            metric_map = {'Overall': 0, 'Visual': 1, 'Demand': 2}
            compare_idx = metric_map[direction]

            val_a = score_a[compare_idx]
            val_b = score_b[compare_idx]

            # 谁在该方向上分数高，谁就是 better
            if val_a >= val_b:
                better_ind, worse_ind = ind_a, ind_b
            else:
                better_ind, worse_ind = ind_b, ind_a

            print(f"  对组 {i//2 + 1}: 方向={direction}, 差异比例 -> 综合:{diff_total:.3f}, 视觉:{diff_visual:.3f}, 需求:{diff_demand:.3f}")

        # 记录代码，保持对应顺序
        worse_code_lst.append(worse_ind['code'])
        better_code_lst.append(better_ind['code'])
        direction_lst.append(direction)

        # 创建异步任务，传入方向信息
        task = generate_single_reflection(client, better_ind, worse_ind, model=model_name, direction=direction)
        tasks.append(task)

    # 并发执行所有 LLM 请求
    hints_list: List[str] = await asyncio.gather(*tasks)

    print("短时反思 Hints 生成完毕。")
    return hints_list, worse_code_lst, better_code_lst, direction_lst





async def single_crossover(
        client: Any,  # 实际应为 AsyncClient
        prompt_content: str,
        system_prompt: str,
        hint: str,
        worse_code: str,
        better_code: str,
        idx: int,
        direction: str,  # 新增参数：反思方向
        op_type: str,  # <--- 新增参数：指定算子类型 ('cx', 'deduce', 'add')
        env: Any,
        evaluate_pop: list,
        toolbox: Any,
        model: str = "qwen-plus",
        current_retry: int = 0,
        max_retries: int = 3,
        last_error: str = None,
        last_code: str = None
) -> str:
    """
    通用算子生成函数：根据 hint 对 worse_code 进行改进。
    特点：LLM生成无超时限制，但算子评估有硬性超时限制（EVALUATION_TIMEOUT_SECONDS）。

    Args:
        direction: 反思方向 (Overall/Visual/Demand)，将添加到改进提示中
    """

    # LLM_TIMEOUT_SECONDS = 60 # 不再用于 wait_for，仅保留作为参考
    EVALUATION_TIMEOUT_SECONDS = 60  # <--- 评估阶段的硬性超时限制

    direction_definitions = {
        "Overall": "Optimize for Pareto Dominance (better trade-off) and Path Diversity (novel routes).",
        "Visual": "Optimize specifically for Objective 1: Path Compactness, Aesthetics, and geometric efficiency.",
        "Demand": "Optimize specifically for Objective 2: Demand Satisfaction, Load Balancing, and Capacity constraints."
    }
    target_goal = direction_definitions.get(direction, "Optimize the operator.")
    # --- 1. 确定算子名称和特定要求 ---
    if op_type == "cx":
        op_name = "crossover"
        func_req_desc = "function must accept (input_ind1, input_ind2, env) and return two valid children in a tuple or list."
    elif op_type == "mt":
        op_name = "mutation"
        func_req_desc = "function must accept (input_ind, env) and return a modified route (list of nodes)."

    # --- 2. 构建 Prompt (区分初次生成和修复模式) ---
    code_str = ""  # 预定义 code_str

    if current_retry == 0:
        prompt_template = readText("./prompts/cx_reflection_prompt.txt")
        reflection_instruction = prompt_template.format(
            worse_code=worse_code,
            better_code=better_code,
            reflection=hint,  # hint 已包含方向前缀
            env_structure=readText("./description/env_desc.txt")
        )
        full_user_prompt = prompt_content + "\n\n--- 改进指令 ---\n" + reflection_instruction
        print(f"[{op_name}] 算子 {idx} 开始生成 (方向: {target_goal})...")
    else:
        print(f"[{op_name}] 算子 {idx} 正在进行第 {current_retry}/{max_retries} 次自我修复...")
        repair_instruction = (
            f"\n\n[ERROR REPORT]\n"
            f"Your previous generated code for {op_name} failed to execute or timed out during the try-run.\n"
            f"Previous Code:\n```python\n{last_code}\n```\n"
            f"Error Message:\n{last_error}\n\n"
            f"[INSTRUCTION]\n"
            f"Please analyze the error and FIX the code.\n"
            f"Requirement: The {func_req_desc}\n"
            f"Ensure you handle node types correctly (e.g., convert np.int64 to int for networkx).\n"
            f"Output ONLY the fixed Python code block."
        )
        full_user_prompt = prompt_content + repair_instruction

    try:
        # 3. LLM 调用 (无硬性超时限制)
        try:
            response = await tracked_chat_create(client, phase="long_reflection",
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': full_user_prompt}
                ],
                temperature=1,
                timeout=300
            )


            new_code_content = response.choices[0].message.content.strip()
            new_code_content = extract_true_response(new_code_content)
            code_str = extract_code_block(new_code_content)

            if not code_str:
                raise ValueError("No code block found in LLM response.")

        except Exception as e:
            # 捕获 LLM 调用的网络错误、连接中断等
            raise Exception(f"LLM Chat Error: {str(e)}")

        # 4. 立即评估 (Try-Run)
        validation_error = None
        op_func = None
        try:
            # A. 编译 (通常很快，无需超时)
            op_func = compile_and_load_code_string(code_str)
            if not op_func:
                raise ValueError("Compilation failed (syntax error or no function found).")

            # B. 运行测试：定义一个同步函数来执行阻塞的评估逻辑
            def _run_evaluation_tests():

                # Case 1: 通用变异算子 (Mutation: 'mt')
                if op_type == 'mt':
                    # 只测前 4 个个体，节省时间
                    test_parents = evaluate_pop[:4] if len(evaluate_pop) >= 4 else evaluate_pop
                    if not test_parents: return

                    for parent in test_parents:
                        p_copy = toolbox.clone(parent)
                        p_copy = convert_individual_nodes_to_int(p_copy)

                        # 直接调用生成的变异函数: mutation(individual, env)
                        # 不再需要 mut_reduce_add 包装器
                        children = op_func(p_copy, env)

                        # 1. 检查返回值类型 (必须是元组)
                        if not isinstance(children, tuple):
                            raise ValueError(f"Mutation operator must return a tuple, got {type(children)}.")

                        # 2. 检查元组长度 (DEAP 变异要求返回长度为 1 的元组)
                        if len(children) != 1:
                            raise ValueError(f"Mutation operator tuple length must be 1, got {len(children)}.")

                        child = children[0]

                        # 3. 检查子代内容
                        if not isinstance(child, list):
                            raise ValueError(f"Mutation child must be a list (Individual), got {type(child)}.")
                        if len(child) == 0:
                            raise ValueError("Mutation operator returned empty child individual.")

                # Case 2: 交叉算子 (Crossover: 'cx')
                elif op_type == 'cx':
                    pairs = list(itertools.combinations(evaluate_pop, 2))
                    test_pairs = pairs[:2] if len(pairs) > 2 else pairs
                    if not test_pairs: return

                    for parent1, parent2 in test_pairs:
                        p1_copy, p2_copy = toolbox.clone(parent1), toolbox.clone(parent2)
                        p1_copy = convert_individual_nodes_to_int(p1_copy)
                        p2_copy = convert_individual_nodes_to_int(p2_copy)

                        # 直接调用生成的交叉函数: cx_operator(ind1, ind2, env)
                        children = op_func(p1_copy, p2_copy, env)

                        # 1. 检查返回值类型
                        if not isinstance(children, (list, tuple)):
                            raise ValueError(f"Crossover operator must return a tuple, got {type(children)}.")

                        # 2. 检查元组长度 (必须是 2)
                        if len(children) != 2:
                            raise ValueError(f"Crossover operator tuple length must be 2, got {len(children)}.")

                        child1, child2 = children

                        # 3. 检查子代内容
                        if not isinstance(child1, list) or not isinstance(child2, list):
                            raise ValueError("Crossover children must be lists.")
                        if len(child1) == 0 or len(child2) == 0:
                            raise ValueError("Crossover operator returned one or more empty children.")

                else:
                    raise ValueError(f"Unknown operator type: {op_type}")

                return True  # 评估成功

            # C. 实际执行：在线程中运行评估逻辑，并设置超时
            await asyncio.wait_for(
                asyncio.to_thread(_run_evaluation_tests),
                timeout=EVALUATION_TIMEOUT_SECONDS
            )

            # D. 如果运行到这里没有报错，说明代码是“可运行的”
            # 注意：这里假设外部循环变量名为 idx 或 index，请根据上下文调整
            print(f"✅ [{op_name}] 算子通过 Try-Run。")
            return code_str

        except asyncio.TimeoutError:
            # 捕获评估超时
            validation_error = f"Evaluation timed out after {EVALUATION_TIMEOUT_SECONDS} seconds."

        except Exception as e:
            # 捕获编译错误、运行时错误、验证失败等
            import traceback
            tb_lines = traceback.format_exc().splitlines()
            # 传递错误类型、错误信息和相关的 traceback 行，帮助 LLM 定位
            relevant_tb = "\n".join(tb_lines[-3:]) if len(tb_lines) >= 3 else str(e)
            validation_error = (
                f"{type(e).__name__}: {str(e)}\n"
                f"Traceback Snippet:\n{relevant_tb}"
            )

        # 5. 错误处理与递归
        if validation_error:
            print(validation_error)
            if current_retry < max_retries:
                # 递归调用进行修复
                return await single_crossover(
                    client, prompt_content, system_prompt, hint, worse_code, better_code, idx, direction,
                    op_type=op_type, env=env, evaluate_pop=evaluate_pop, toolbox=toolbox,
                    model=model,
                    current_retry=current_retry + 1,
                    max_retries=max_retries,
                    last_error=validation_error,
                    last_code=code_str,

                )
            else:
                print(f"❌ [{op_name}] 算子 {idx} 超过最大重试次数，放弃。最后错误: {validation_error}")
                return f"# Generation Failed after {max_retries} retries.\n# Error: {validation_error}\n{code_str}"

    except Exception as e:
        # 处理 LLM 调用失败（网络、连接错误）
        error_message = str(e)
        if current_retry < max_retries:
            return await single_crossover(
                client, prompt_content, system_prompt, hint, worse_code, better_code, idx, direction,
                op_type=op_type, env=env, evaluate_pop=evaluate_pop, toolbox=toolbox,
                model=model,
                current_retry=current_retry + 1,
                max_retries=max_retries,
                last_error=error_message,
                last_code="LLM Chat Error" if current_retry == 0 else last_code
            )
        else:
            print(f"❌ [{op_name}] 算子 {idx} 系统错误导致放弃: {error_message}")
            return f"# System Error during generation of {op_name}: {error_message}"
async def crossover(
        client: AsyncClient,
        toolbox: Any,
        population: List[Dict[str, Any]],
        hints: List[str],
        worse_codes: List[str],
        better_codes: List[str],
        directions: List[str],  # 新增参数：方向列表
        prompt_content: str,
        system_prompt: str,
        op_type: str,
        model_name: str = "qwen-plus"
) -> List[Dict[str, Any]]:


    if len(hints) != len(worse_codes):
        print("警告：Hints 数量与 Worse Code 数量不匹配，无法进行交叉操作。")
        return population

    tasks = []

    # 获取当前种群的最大索引，用于新个体的命名
    current_max_idx = max(ind.get('idx', -1) for ind in population) if population else -1

    print(f"开始并行生成 {len(hints)} 个交叉/变异算子...")

    for i, (hint, worse_code, better_code, direction) in enumerate(zip(hints, worse_codes, better_codes, directions)):
        new_idx = current_max_idx + i + 1

        task = single_crossover(
            client,
            prompt_content,
            system_prompt,
            hint,
            worse_code,
            better_code,
            new_idx,
            direction,  # 传递方向参数
            op_type=op_type,
            env = env,
            evaluate_pop=evaluate_pop,
            toolbox=toolbox,
            model=model_name
        )
        tasks.append(task)

    # 并发执行所有 LLM 请求
    new_code_results: List[str] = await asyncio.gather(*tasks)

    new_population_segment: List[Dict[str, Any]] = []

    # 处理结果并构建新个体
    successful_generations = 0
    for i, new_code in enumerate(new_code_results):
        if new_code:
            new_idx = current_max_idx + i + 1
            new_individual = {
                "idx": new_idx,
                "code": new_code,
                "score": -1  # 标记为待评估
            }
            new_population_segment.append(new_individual)
            successful_generations += 1

    print(f"成功生成了 {successful_generations} 个新的交叉/变异算子。")

    # 合并种群
    new_population = population + new_population_segment
    print(f"新种群规模: {len(new_population)}")

    return new_population





def gen_long_term_reflection(
        client: OpenAI,  # 注意：现在需要传入 OpenAI (同步) 实例
        pre_long_term_reflection: str,
        hints: List[str],
        system_prompt: str,
        lt_reflection_prompt_path: str = "./prompts/lt_reflection_prompt.txt",
        model_name: str = "qwen-plus"
) -> str:


    lt_reflection_template = readText(lt_reflection_prompt_path)

    # 2. 格式化 Hints 列表为字符串
    hints_str = "\n- " + "\n- ".join(hints) if hints else "no hint。"

    # 3. 格式化用户 Prompt
    user_prompt = lt_reflection_template.format(
        prior_reflection=pre_long_term_reflection,
        new_reflection=hints_str
    )

    print("正在调用 LLM 生成长期反思...")

    try:

        response = tracked_chat_create(client, phase="long_reflection",
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        raw_content = response.choices[0].message.content.strip()

        # 5. 提取真正的回复
        new_long_term_reflection = extract_true_response(raw_content)

        print("长期反思生成完毕。")
        return new_long_term_reflection

    except Exception as e:
        print(f"生成长期反思失败: {e}")
        return pre_long_term_reflection


import concurrent.futures
from func_timeout import func_timeout, FunctionTimedOut
import traceback


def mutation(
        client: Any,
        elitist_individual: Dict[str, Any],
        long_term_reflection: str,
        base_prompt_content: str,
        system_prompt: str,
        env: Any,
        evaluate_pop: list,
        toolbox: Any,
        op_type: str,
        mutate_prompt_path: str = "./prompts/mt_reflection_prompt.txt",
        model_name: str = "qwen-plus",
        direction_hint: str = ""  # 新增参数：方向提示
) -> Dict[str, Any]:
    """
    同步函数：基于长期反思对精英算子进行变异/改进。
    修复版：增加了对【生成的代码执行过程】的超时监控，防止死循环卡死主进程。

    Args:
        direction_hint: 方向提示 (如 "[Focus on Visual optimization] ")，将添加到 prompt 中
    """

    # 增加导入（如果外部未导入）
    try:
        from func_timeout import func_timeout, FunctionTimedOut
    except ImportError:
        raise ImportError("Please run 'pip install func_timeout' to handle code execution timeouts.")

    print(f"\n正在调用 LLM 生成精英算子 (Type: {op_type})...")
    LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # 1. 准备基础 Prompt
    try:
        mutate_template = readText(mutate_prompt_path)
    except FileNotFoundError:
        print(f"错误: 找不到变异模板文件: {mutate_prompt_path}")
        return {"idx": -1, "code": "Error: Mutate template not found", "score": -1}

    initial_user_prompt = mutate_template.format(
        reflection=long_term_reflection,
        elitist_code=elitist_individual["code"],
        env_structure=readText("./description/env_desc.txt")
    )

    # 添加方向提示
    if direction_hint:
        initial_user_prompt = f"{direction_hint}\n\n{initial_user_prompt}"
        print(f"方向提示: {direction_hint}")

    # 辅助函数：LLM 调用（已接入追踪器，phase=crossover_gen）
    def _run_client_chat(current_model, messages):
        return tracked_chat_create(client, phase="crossover_gen",
            model=current_model,
            messages=messages,
            temperature=1,
            timeout=300
        )

    # 2. 递归生成与修复函数
    def _recursive_generation(
            current_retry: int = 0,
            max_retries: int = 3,
            last_error: str = None,
            last_code: str = None
    ) -> str:

        LLM_GEN_TIMEOUT = 300  # 生成代码的超时时间
        CODE_EXEC_TIMEOUT = 10  # 运行代码的超时时间 (Try-Run 不应超过 10 秒)

        # A. 确定描述
        if op_type == "cx":
            op_name = "crossover"
            func_req_desc = "function must accept (input_ind1, input_ind2, env) and return two valid children in a tuple."
        elif op_type == "mt":
            op_name = "mutation"
            func_req_desc = "function must accept (input_ind, env) and return a modified route individual in a tuple `(new_ind, )`."

        # B. 构建 Prompt
        if current_retry == 0:
            final_user_prompt = initial_user_prompt
        else:
            print(f"精英算子正在进行第 {current_retry}/{max_retries} 次自我修复...")
            repair_instruction = (
                f"\n\n[ERROR REPORT]\n"
                f"Your previous generated code for {op_name} failed to execute or timed out during the try-run.\n"
                f"Previous Code:\n```python\n{last_code}\n```\n"
                f"Error Message:\n{last_error}\n\n"
                f"[INSTRUCTION]\n"
                f"Please analyze the error and FIX the code.\n"
                f"1. Check for infinite loops (while True).\n"
                f"2. Check for unconnected graph errors (NetworkXNoPath).\n"
                f"3. Requirement: The {func_req_desc}\n"
                f"Output ONLY the fixed Python code block."
            )
            final_user_prompt = initial_user_prompt + repair_instruction

        try:
            # C. 调用 LLM (生成阶段超时控制)
            code_str = ""
            chat_messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': final_user_prompt}
            ]

            future = LLM_EXECUTOR.submit(_run_client_chat, model_name, chat_messages)

            try:
                response = future.result(timeout=LLM_GEN_TIMEOUT)
                raw_content = response.choices[0].message.content.strip()
                raw_content = extract_true_response(raw_content)
                code_str = extract_code_block(raw_content)
                if not code_str:
                    raise ValueError("No code block found in LLM response.")
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(f"LLM generation timed out after {LLM_GEN_TIMEOUT} seconds.")

            # D. 立即评估 (Try-Run 阶段超时控制)
            # D.1 编译
            op_func = compile_and_load_code_string(code_str)
            if not op_func:
                raise ValueError("Compilation failed.")

            # D.2 定义执行逻辑
            def _execute_try_run():
                if op_type == 'mt':
                    test_parents = evaluate_pop[:4] if len(evaluate_pop) >= 4 else evaluate_pop
                    for parent in test_parents:
                        p_copy = convert_individual_nodes_to_int(toolbox.clone(parent))
                        children = op_func(p_copy, env)
                        if not isinstance(children, tuple) or len(children) != 1:
                            raise ValueError(f"Mutation must return a tuple of length 1, got {type(children)}")
                        if not isinstance(children[0], list):
                            raise ValueError("Mutation child must be a list.")

                elif op_type == 'cx':
                    pairs = list(itertools.combinations(evaluate_pop, 2))
                    test_pairs = pairs[:2] if len(pairs) > 2 else pairs
                    if test_pairs:
                        for p1, p2 in test_pairs:
                            p1_c = convert_individual_nodes_to_int(toolbox.clone(p1))
                            p2_c = convert_individual_nodes_to_int(toolbox.clone(p2))
                            children = op_func(p1_c, p2_c, env)
                            if not isinstance(children, tuple) or len(children) != 2:
                                raise ValueError("Crossover must return a tuple of length 2.")
                return True

            # D.3 执行并监控超时
            try:
                func_timeout(CODE_EXEC_TIMEOUT, _execute_try_run)
            except FunctionTimedOut:
                raise TimeoutError(
                    f"Generated code execution timed out after {CODE_EXEC_TIMEOUT}s (Possible infinite loop).")
            except Exception as e:
                raise e  # 抛出其他运行时错误

            # E. 成功
            print(f"精英算子 ({op_type}) 生成并验证成功。")
            return code_str

        except (Exception, TimeoutError) as e:
            # F. 失败处理与递归
            tb_lines = traceback.format_exc().splitlines()
            error_msg = f"{type(e).__name__}: {str(e)}\nTraceback Snippet:\n{tb_lines[-1]}"

            if current_retry < max_retries:
                return _recursive_generation(
                    current_retry + 1, max_retries, error_msg,
                    code_str if code_str else "No code generated"
                )
            else:
                print(f"精英算子生成失败 (Max Retries): {error_msg}")
                return f"# Final Error: {error_msg}"

    # 3. 执行
    final_code = _recursive_generation()

    # 4. 构建结果
    return {
        "idx": -1,
        "code": final_code,
        "score": -1
    }


async def llm_gen_cx_agent(env, evaluate_pop, toolbox, summary_detail: str = "", code: Dict[str, Any] = None,
                           is_init: bool = False):
    # ================= 配置区域 =================
    # 建议通过环境变量获取，或在此处填入您的阿里云 API Key
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus"
    # 初始化客户端
    sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    async_client = AsyncOpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

    # T12: 追踪器初始化
    tracker = LLMTracker.get_instance()
    evolution_id = time.strftime("%Y%m%d_%H%M%S")
    evolution_out_dir = os.environ.get(
        "LEHHA_EVOLUTION_DIR",
        os.path.join("./results/LEHHA/evolution", evolution_id)
    )
    os.makedirs(evolution_out_dir, exist_ok=True)
    os.makedirs(os.path.join(evolution_out_dir, "elitists"), exist_ok=True)
    tracker.start_evolution(evolution_id, MODEL_NAME)
    # 让后续函数体可访问
    _cx_evolution_out_dir = evolution_out_dir

    if not is_init:
        # 确保 code 参数有效
        if code is None:
            print("错误：非初始化模式下，'code' 参数（原始精英算子）不能为空。")
            return None

        # 1. 准备 Prompt
        system_prompt = readText("./prompts/system_prompt.txt")
        prompt_template = readText("./prompts/improve_op_in_loop.txt")
        prompt = prompt_template.format(
            code=code["code"],
            summary_detail=summary_detail
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        print(f"正在调用{MODEL_NAME}基于反思改进算子...")

        # 2. 调用 API (同步方式)
        response = tracked_chat_create(sync_client, phase="crossover_gen",
            model=MODEL_NAME,
            messages=messages
        )

        # 3. 解析响应 (OpenAI 格式)
        result = response.choices[0].message.content.strip()
        result = extract_true_response(result)
        new_code = extract_code_block(result)

        # 后续逻辑保持不变
        original_ind = {
            "idx": code.get("idx", -1),
            "code": code["code"],
            "score": code.get("score", -1)
        }
        new_ind = {"idx": -2, "code": new_code, "score": -1}
        population_to_evaluate = [new_ind]
        evaluated_population = evaluate_operator(population_to_evaluate, evaluate_pop, toolbox, env, op_type="cx")

        final_new_ind = evaluated_population[0]
        original_score = original_ind.get('score', -float('inf'))
        new_score = final_new_ind.get('score', -float('inf'))

        if new_score > original_score:
            print(f"改进成功！新算子得分 ({new_score:.4f}) 优于原算子 ({original_score:.4f})。")
            return final_new_ind
        else:
            print(f"改进失败或无提升。原算子得分 ({original_score:.4f}) 优于或等于新算子 ({new_score:.4f})。")
            print("返回原算子代码。")
            return original_ind


    else:

        NUM_OPERATORS = 48
        NUM_GEN = 5
        elitist = None
        long_term_reflection = ""
        # ================= 1. 准备工作 =================

        print("正在调用启发式智能体生成规则...")
        rule = llm_heuristic_agent(summary_detail, target="cx")
        print(rule)
        # rule = "None"
        print("正在预读取 Prompt 文件...")
        system_prompt = readText("./prompts/system_prompt.txt")
        user_prompt_template = readText("./prompts/gen_cx_prompt.txt")
        problem_desc = readText("./description/description.txt")
        default_cx_operator = readText("./description/cx_default.txt")
        env_structure = readText("./description/env_desc.txt")
        base_prompt_content = user_prompt_template.format(
            problem_desc=problem_desc,
            heuristic_rule=rule,
            default_crossover_operator=default_cx_operator,
            env_structure=env_structure
        )

        # ================= 2. 初始种群生成 (Async) =================

        print(f"开始并行生成 {NUM_OPERATORS} 个交叉算子...")

        tasks: List[Awaitable[str]] = []

        for i in range(1, NUM_OPERATORS + 1):
            tasks.append(
                generate_single_operator(
                    async_client,
                    env,
                    base_prompt_content,
                    system_prompt,
                    i,
                    evaluate_pop=evaluate_pop,
                    toolbox=toolbox,
                    op_type="cx",
                    model_name=MODEL_NAME
                )

            )

        results: List[str] = await asyncio.gather(*tasks)
        population: List[Dict[str, Any]] = [
            {"idx": i, "code": code_str, "score": -1}
            for i, code_str in enumerate(results)
        ]
        print("初始算子生成完毕。")

        # 保存初始代码
        for i, ind in enumerate(population):
            with open(f"./temp/crossover_init_operator{ind['idx']}.py", "w", encoding="utf-8") as f:
                f.write(ind['code'])
        # ================= 3. 第一轮评估与初始化精英 =================
        print("正在进行初始评估...")
        population = evaluate_operator(population, evaluate_pop, toolbox, env, op_type="cx", seed = 42)

        # 初始化三个精英
        # 按综合分排序
        population.sort(key=lambda ind: ind.get('score', [-float('inf')] * 3)[0] if isinstance(ind.get('score'), (tuple, list)) else ind.get('score', -float('inf')), reverse=True)

        # 初始化三个精英
        elitist_overall = population[0].copy()
        elitist_visual = max(population, key=lambda ind: ind.get('score', [-float('inf')] * 3)[1] if isinstance(ind.get('score'), (tuple, list)) else -float('inf')).copy()
        elitist_demand = max(population, key=lambda ind: ind.get('score', [-float('inf')] * 3)[2] if isinstance(ind.get('score'), (tuple, list)) else -float('inf')).copy()

        print(f"初始精英确立 - 综合:{elitist_overall['score'][0]}, 视觉:{elitist_visual['score'][1]}, 需求:{elitist_demand['score'][2]}")
        # --- 配置日志文件路径 ---
        log_filename = "./results/cx_evolution_history.csv"

        # 初始化文件：写入表头
        with open(log_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Elite_Overall", "Elite_Visual", "Elite_Demand",
                             "Avg_Overall", "Avg_Visual", "Avg_Demand"])

        print(f"日志文件已创建: {log_filename}")
        history = []

        # ================= 4. 进化主循环 =================
        for i in range(NUM_GEN):
            print(f"\n=== 正在进行第 {i + 1} / {NUM_GEN} 代进化 ===")
            # --- 4.1 Short-term Reflection (本地模型，开支小) ---
            # 打乱后，强弱算子随机配对，能产生更强的"梯度"，提升 Reflection 质量。
            random.shuffle(population)
            print("正在进行短时反思...")
            hints, worse_codes, better_codes, directions = await short_term_reflection(
                async_client,
                population,
                model_name=MODEL_NAME
            )

            # --- 4.2 Crossover (Qwen) ---

            print("正在交叉生成新算子...")

            population_after_cx = await crossover(
                async_client,
                toolbox,
                population,
                hints,
                worse_codes,
                better_codes,
                directions,  # 传递方向列表
                problem_desc,
                system_prompt,
                op_type="cx",
                model_name=MODEL_NAME
            )

            # --- 4.3 评估交叉后代 ---

            print("正在评估交叉后的种群...")
            # 这里的评估是必要的，给新生成的子代打分
            population_after_cx = evaluate_operator(population_after_cx, evaluate_pop, toolbox, env, op_type="cx",seed = i)
            # --- 4.4 选择策略 (修复引用 + 语义去重 + 三分数精英更新) ---
            mixed_population = population + population_after_cx

            # a. 更新三个精英
            # 辅助函数：安全获取分数
            def get_score_safe(ind, idx):
                score = ind.get('score')
                if isinstance(score, (tuple, list)) and len(score) > idx:
                    return score[idx]
                return -float('inf')

            current_best_overall = max(mixed_population, key=lambda ind: get_score_safe(ind, 0))
            current_best_visual = max(mixed_population, key=lambda ind: get_score_safe(ind, 1))
            current_best_demand = max(mixed_population, key=lambda ind: get_score_safe(ind, 2))

            # 更新综合分精英
            if get_score_safe(current_best_overall, 0) > get_score_safe(elitist_overall, 0):
                print(f"  >>> [综合] 发现新精英！{get_score_safe(elitist_overall, 0):.4f} -> {get_score_safe(current_best_overall, 0):.4f}")
                elitist_overall = current_best_overall.copy()

            # 更新视觉分精英
            if get_score_safe(current_best_visual, 1) > get_score_safe(elitist_visual, 1):
                print(f"  >>> [视觉] 发现新精英！{get_score_safe(elitist_visual, 1):.4f} -> {get_score_safe(current_best_visual, 1):.4f}")
                elitist_visual = current_best_visual.copy()

            # 更新需求分精英
            if get_score_safe(current_best_demand, 2) > get_score_safe(elitist_demand, 2):
                print(f"  >>> [需求] 发现新精英！{get_score_safe(elitist_demand, 2):.4f} -> {get_score_safe(current_best_demand, 2):.4f}")
                elitist_demand = current_best_demand.copy()

            # b. 构建下一代 (加入去重逻辑 + 精英保护)
            next_generation = []
            seen_codes = set()

            # 1. 精英保护：保护三个精英（自动去重）
            elitists_to_protect = [elitist_overall, elitist_visual, elitist_demand]
            for elit in elitists_to_protect:
                if elit['code'] not in seen_codes:
                    next_generation.append(elit.copy())
                    seen_codes.add(elit['code'])

            # 2. 锦标赛选择填补剩余空位
            needed_count = NUM_OPERATORS - len(next_generation)
            tournament_size = 3

            # 使用 while 循环，确保填满且不重复
            attempts = 0
            max_attempts = needed_count * 5  # 防止死循环

            while len(next_generation) < NUM_OPERATORS and attempts < max_attempts:
                attempts += 1
                # 随机抽取 k 个
                candidates = random.sample(mixed_population, min(len(mixed_population), tournament_size))
                # 按综合分选择优胜者
                winner = max(candidates, key=lambda ind: get_score_safe(ind, 0))

                # 【核心修正】：不仅断开引用，还要检查代码内容是否已存在
                if winner['code'] not in seen_codes:
                    ind_new = winner.copy()  # 内存去重
                    next_generation.append(ind_new)
                    seen_codes.add(winner['code'])  # 语义去重

            # 如果尝试了很多次还没填满（说明种群同质化严重），则允许重复填满
            while len(next_generation) < NUM_OPERATORS:
                candidates = random.sample(mixed_population, min(len(mixed_population), tournament_size))
                # 按综合分选择优胜者
                winner = max(candidates, key=lambda ind: get_score_safe(ind, 0))
                ind_new = winner.copy()
                next_generation.append(ind_new)
                # 这里不再 check seen_codes，为了保证种群数量

            # ================= 5. 长期反思与变异 =================

            # --- 5.1 Long-term Reflection (Sync, 本地模型) ---
            pre_long_term_reflection = long_term_reflection if long_term_reflection != "" else ""
            long_term_reflection = gen_long_term_reflection(
                sync_client,
                pre_long_term_reflection,
                hints,
                system_prompt
            )

            # print(f"长期反思更新: {long_term_reflection[:50]}...")

            # --- 5.2 Mutation (Qwen, 对三个精英分别进行变异) ---

            print("正在进行变异...")


            # 对三个精英分别在三个方向上进行变异 (3 * 3 = 9 次变异)
            mutated_individuals = []

            # 定义精英来源
            elitist_sources = [
                ("OverallElit", elitist_overall),
                ("VisualElit", elitist_visual),
                ("DemandElit", elitist_demand)
            ]

            direction_prompts = {
                "Overall": "Improve Pareto Dominance and Population Diversity. Try to find a better trade-off between objectives.",
                "Visual": "Focus specifically on optimizing Objective 1 (Visual Compactness & Path Aesthetics). It is acceptable to sacrifice some Demand satisfaction.",
                "Demand": "Focus specifically on optimizing Objective 2 (Demand Satisfaction & Load Balancing). It is acceptable to sacrifice some Visual compactness."
            }

            target_directions = ["Overall", "Visual", "Demand"]

            for source_name, elit_ind in elitist_sources:
                # 安全检查：防止某个维度的精英不存在
                if not elit_ind:
                    continue

                for dir_name in target_directions:
                    # 获取详细的提示词
                    detailed_hint = direction_prompts.get(dir_name, f"Focus on {dir_name}")

                    print(f"  变异源: {source_name} -> 目标方向: {dir_name}")

                    mutated_ind = mutation(
                        sync_client,
                        elit_ind,
                        long_term_reflection,
                        base_prompt_content,
                        system_prompt,
                        env=env,
                        evaluate_pop=evaluate_pop,
                        toolbox=toolbox,
                        op_type="cx",
                        model_name=MODEL_NAME,
                        # --- 【修改点】传入详细的提示词 ---
                        direction_hint=f"[Mutation Goal: {detailed_hint}] "
                    )

                    # 修复：只匹配 mutation 失败分支的精确前缀（"# Final Error:"），
                    # 不再用 "Error" not in code 的子串匹配，避免误杀含
                    # `except ValueError` / `raise NetworkXError` 等合理代码的算子
                    if mutated_ind and not mutated_ind.get("code", "").startswith("# Final Error:"):
                        mutated_individuals.append(mutated_ind)

                        # 文件名依然使用简短的 dir_name，保持整洁
                        file_name = f"temp/cx_mutated_{source_name.lower()}_to_{dir_name.lower()}.py"
                        with open(file_name, "w", encoding="utf-8") as f:
                            f.write(mutated_ind["code"])

            # 将所有变异个体加入种群
            for mutated_ind in mutated_individuals:
                next_generation.append(mutated_ind)
            print(f"共生成 {len(mutated_individuals)} 个变异算子。")
            # ================= 6. 最终评估与收缩 =================
            print("正在进行本代最终全量评估 (触发加权平滑)...")
            evaluated_final_pop = evaluate_operator(next_generation, evaluate_pop, toolbox, env, op_type="cx",
                                                    seed=i)

            # 按综合分排序
            evaluated_final_pop.sort(key=lambda ind: get_score_safe(ind, 0), reverse=True)

            # 更新三个精英
            best_overall = evaluated_final_pop[0]
            best_visual = max(evaluated_final_pop, key=lambda ind: get_score_safe(ind, 1))
            best_demand = max(evaluated_final_pop, key=lambda ind: get_score_safe(ind, 2))

            if get_score_safe(best_overall, 0) > get_score_safe(elitist_overall, 0):
                print(f"  >>> [综合] 最终精英突破！{get_score_safe(elitist_overall, 0):.4f} -> {get_score_safe(best_overall, 0):.4f}")
            if get_score_safe(best_visual, 1) > get_score_safe(elitist_visual, 1):
                print(f"  >>> [视觉] 最终精英突破！{get_score_safe(elitist_visual, 1):.4f} -> {get_score_safe(best_visual, 1):.4f}")
            if get_score_safe(best_demand, 2) > get_score_safe(elitist_demand, 2):
                print(f"  >>> [需求] 最终精英突破！{get_score_safe(elitist_demand, 2):.4f} -> {get_score_safe(best_demand, 2):.4f}")

            elitist_overall = best_overall.copy()
            elitist_visual = best_visual.copy()
            elitist_demand = best_demand.copy()

            # 截断回 NUM_OPERATORS (丢弃表现最差的，保持种群大小恒定)
            population = evaluated_final_pop[:NUM_OPERATORS]

            # 记录数据 - 计算三个维度的平均分
            valid_scores = [ind['score'] for ind in population if isinstance(ind.get('score'), (tuple, list)) and ind['score'][0] > -1000]
            if valid_scores:
                avg_overall = statistics.mean([s[0] for s in valid_scores])
                avg_visual = statistics.mean([s[1] for s in valid_scores])
                avg_demand = statistics.mean([s[2] for s in valid_scores])
            else:
                avg_overall = avg_visual = avg_demand = -1

            print(f"第 {i + 1} 代结束。")
            print(f"  精英 - 综合:{get_score_safe(elitist_overall, 0):.4f}, 视觉:{get_score_safe(elitist_visual, 1):.4f}, 需求:{get_score_safe(elitist_demand, 2):.4f}")
            print(f"  平均 - 综合:{avg_overall:.4f}, 视觉:{avg_visual:.4f}, 需求:{avg_demand:.4f}")

            history.append((elitist_overall['score'], elitist_visual['score'], elitist_demand['score'],
                            (avg_overall, avg_visual, avg_demand)))
            # 记录本代 cx 算子池精英分与平均分（统一持久化到 temp/operator_performance_iter.json）
            _operator_performance_recorder.record(
                "cx", i + 1,
                (get_score_safe(elitist_overall, 0), get_score_safe(elitist_visual, 1), get_score_safe(elitist_demand, 2)),
                (avg_overall, avg_visual, avg_demand)
            )
            try:
                with open(log_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 写入一行: [代数, 三个精英分, 三个平均分]
                    writer.writerow([i + 1,
                                     f"{get_score_safe(elitist_overall, 0):.4f}",
                                     f"{get_score_safe(elitist_visual, 1):.4f}",
                                     f"{get_score_safe(elitist_demand, 2):.4f}",
                                     f"{avg_overall:.4f}",
                                     f"{avg_visual:.4f}",
                                     f"{avg_demand:.4f}"])
            except Exception as e:
                print(f"写入日志失败: {e}")
        # ================= 7. 结束工作 =================
        print(f"\n流程结束。")
        print(f"综合精英 ID: {elitist_overall.get('idx', 'unknown')}, Score: {elitist_overall['score']}")
        print(f"视觉精英 ID: {elitist_visual.get('idx', 'unknown')}, Score: {elitist_visual['score']}")
        print(f"需求精英 ID: {elitist_demand.get('idx', 'unknown')}, Score: {elitist_demand['score']}")

        # 保存三个精英算子（T12: 双路径保存 temp/ 与 evolution_out_dir/elitists/）
        _cx_elitist_pairs = [
            ("cx_elitist_overall.py", elitist_overall),
            ("cx_elitist_visual.py", elitist_visual),
            ("cx_elitist_demand.py", elitist_demand),
        ]
        for fname, elitist_obj in _cx_elitist_pairs:
            for target_dir in ["temp", os.path.join(_cx_evolution_out_dir, "elitists")]:
                os.makedirs(target_dir, exist_ok=True)
                with open(os.path.join(target_dir, fname), "w", encoding="utf-8") as f:
                    f.write(elitist_obj['code'])

        print("精英算子已保存:")
        print(f"  - 综合: temp/cx_elitist_overall.py (Score: {elitist_overall['score']})")
        print(f"  - 视觉: temp/cx_elitist_visual.py (Score: {elitist_visual['score']})")
        print(f"  - 需求: temp/cx_elitist_demand.py (Score: {elitist_demand['score']})")

        # T12: 追踪器导出
        tracker.end_evolution()
        tracker.export_json(_cx_evolution_out_dir)
        tracker.export_csv(_cx_evolution_out_dir)
        print(f"[追踪] LLM 使用统计已保存到 {_cx_evolution_out_dir}/llm_usage.json")

        return {
            "overall": elitist_overall,
            "visual": elitist_visual,
            "demand": elitist_demand
        }

async def llm_gen_mt_agent(env,evaluate_pop, toolbox, summary_detail: str = "", code: List[Dict[str, Any]] = None,
                           is_init: bool = False):
    # ================= 配置区域 =================
    # 建议通过环境变量获取，或在此处填入您的阿里云 API Key
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus"
    # 初始化客户端
    sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    async_client = AsyncOpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

    # T12: 追踪器初始化（mt 与 cx 共享同一单例；若 cx 已 start，此处保留状态）
    tracker = LLMTracker.get_instance()
    if tracker.evolution_id is None:
        # 单独运行 mt_agent 时初始化
        _mt_evolution_id = time.strftime("%Y%m%d_%H%M%S")
        _mt_evolution_out_dir = os.environ.get(
            "LEHHA_EVOLUTION_DIR",
            os.path.join("./results/LEHHA/evolution", _mt_evolution_id)
        )
        os.makedirs(_mt_evolution_out_dir, exist_ok=True)
        os.makedirs(os.path.join(_mt_evolution_out_dir, "elitists"), exist_ok=True)
        tracker.start_evolution(_mt_evolution_id, MODEL_NAME)
    else:
        _mt_evolution_out_dir = os.environ.get(
            "LEHHA_EVOLUTION_DIR",
            os.path.join("./results/LEHHA/evolution", tracker.evolution_id)
        )
        os.makedirs(os.path.join(_mt_evolution_out_dir, "elitists"), exist_ok=True)

    if not is_init:
        # 确保 code 参数有效且包含两个算子
        if code is None or len(code) < 2:
            print("错误：非初始化模式下，'code' 参数必须是包含 mut_deduce 和 mut_add 两个算子的列表。")
            # 返回原始传入的，以防外部逻辑崩溃
            return code if code is not None else []

        # 分别获取两个原始精英算子
        original_deduce_op = code[0]
        original_add_op = code[1]
        # 1. 使用已初始化的客户端和变量
        client = sync_client  # 使用 OpenAI 客户端
        model_name = MODEL_NAME  # 使用 qwen-plus
        system_prompt = readText("./prompts/system_prompt.txt")
        prompt_template = readText("./prompts/improve_op_in_loop.txt")
        best_operators = []
        for i in range(2):
            prompt = prompt_template.format(
                code=code[i]["code"],
                summary_detail=summary_detail
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            print("正在调用 LLM 基于反思改进算子...")
            response = tracked_chat_create(client, phase="mutation",
                model=model_name,
                messages=messages
            )
            result = response.choices[0].message.content.strip()
            result = extract_true_response(result)
            new_code = extract_code_block(result)
            original_ind = {
                "idx": code[i].get("idx", -1),
                "code": code[i]["code"],
                "score": code[i].get("score", -1)
            }
            new_ind = {"idx": -2, "code": new_code, "score": -1}
            # 封装为包含新旧个体的列表，确保都在同一评估集上运行
            population_to_evaluate = [new_ind]
            # 评估原个体和新个体
            if i == 0:
                evaluated_population = evaluate_operator(population_to_evaluate, evaluate_pop, toolbox,env,op_type="deduce")
            else:
                evaluated_population = evaluate_operator(population_to_evaluate, evaluate_pop, toolbox, env,
                                                         op_type="add")
            # 比较分数并返回最佳结果
            final_new_ind = evaluated_population[0]
            original_score = original_ind.get('score', -float('inf'))
            new_score = final_new_ind.get('score', -float('inf'))

            if new_score > original_score:
                print(f"改进成功！新算子得分 ({new_score:.4f}) 优于原算子 ({original_score:.4f})。")
                best_operators.append(final_new_ind)
            else:
                print(f"改进失败或无提升。原算子得分 ({original_score:.4f}) 优于或等于新算子 ({new_score:.4f})。")
                print("返回原算子代码。")
                # 确保返回的是最新的评估分数个体
                best_operators.append(original_ind)
        return best_operators
    else:
        # ==========================================
        #               变异算子 (MT) 进化主循环
        # ==========================================
        NUM_OPERATORS = 48
        NUM_GEN = 5
        elitist = None
        long_term_reflection = ""
        operator_type = "mt"  # 统一标识符

        # ================= 1. 准备工作 =================

        print("正在调用启发式智能体生成规则...")
        # 注意：这里的 target="mt" 会触发你最新的通用提示词逻辑
        rule = llm_heuristic_agent(summary_detail, target="mt")

        print("正在预读取 Prompt 文件...")
        system_prompt = readText("./prompts/system_prompt.txt")
        user_prompt_template = readText("./prompts/gen_mt_prompt.txt")  # 确保这个文件存在
        problem_desc = readText("./description/description.txt")

        # 【注意】这里需要一个默认的变异算子作为 Few-shot 示例
        # 建议使用之前的 mt_deduce_default.txt 或 mt_default.txt 中较好的一个，或者合并的
        default_mt_operator = readText("./description/mt_default.txt")

        env_structure = readText("./description/env_desc.txt")

        base_prompt_content = user_prompt_template.format(
            problem_desc=problem_desc,
            heuristic_rule=rule,
            default_mutation_operator=default_mt_operator,
            env_structure=env_structure
        )

        # ================= 2. 初始种群生成 (Async) =================

        print(f"开始并行生成 {NUM_OPERATORS} 个变异算子...")

        tasks: List[Awaitable[str]] = []

        for i in range(1, NUM_OPERATORS + 1):
            tasks.append(
                generate_single_operator(
                    async_client,
                    env,
                    base_prompt_content,
                    system_prompt,
                    i,
                    evaluate_pop=evaluate_pop,
                    toolbox=toolbox,
                    op_type=operator_type,  # "mt"
                    model_name=MODEL_NAME
                )
            )

        results: List[str] = await asyncio.gather(*tasks)

        population: List[Dict[str, Any]] = [
            {"idx": i, "code": code_str, "score": -1}
            for i, code_str in enumerate(results)
        ]
        print("初始变异算子生成完毕。")

        # 保存初始代码
        for i, ind in enumerate(population):
            # 简单过滤明显错误的生成
            if "def mutation" not in ind['code'] and "def mt_operator" not in ind['code']:
                # 根据你的 prompt 约定的函数名调整，通常是 mutation
                pass
            with open(f"./temp/mt_init_operator{ind['idx']}.py", "w", encoding="utf-8") as f:
                f.write(ind['code'])

        # ================= 3. 第一轮评估与初始化精英 =================
        print("正在进行初始评估...")
        # 注意：evaluate_operator 内部需要处理 op_type="mt" 的逻辑 (distance, evaluate等)
        population = evaluate_operator(population, evaluate_pop, toolbox, env, op_type=operator_type, seed=42)

        # 初始化三个精英
        # 按综合分排序
        population.sort(key=lambda ind: ind.get('score', [-float('inf')] * 3)[0] if isinstance(ind.get('score'), (tuple, list)) else ind.get('score', -float('inf')), reverse=True)

        # 初始化三个精英
        elitist_overall = population[0].copy()
        elitist_visual = max(population, key=lambda ind: ind.get('score', [-float('inf')] * 3)[1] if isinstance(ind.get('score'), (tuple, list)) else -float('inf')).copy()
        elitist_demand = max(population, key=lambda ind: ind.get('score', [-float('inf')] * 3)[2] if isinstance(ind.get('score'), (tuple, list)) else -float('inf')).copy()

        print(f"初始精英确立 - 综合:{elitist_overall['score'][0]}, 视觉:{elitist_visual['score'][1]}, 需求:{elitist_demand['score'][2]}")
        # --- 配置日志文件路径 ---
        log_filename = "./results/mt_evolution_history.csv"

        # 初始化文件：写入表头
        # "w" 模式会覆盖旧文件，如果你想保留旧历次运行记录，可以改文件名或加时间戳
        with open(log_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Elite_Overall", "Elite_Visual", "Elite_Demand",
                             "Avg_Overall", "Avg_Visual", "Avg_Demand"])

        print(f"日志文件已创建: {log_filename}")
        history = []

        # ================= 4. 进化主循环 =================
        for i in range(NUM_GEN):
            print(f"\n=== 正在进行第 {i + 1} / {NUM_GEN} 代进化 (Type: {operator_type}) ===")

            # --- 4.1 Short-term Reflection ---
            # 打乱后，强弱算子随机配对，能产生更强的"梯度"，提升 Reflection 质量。
            random.shuffle(population)
            print("正在进行短时反思...")
            hints, worse_codes, better_codes, directions = await short_term_reflection(
                async_client,
                population,
                model_name=MODEL_NAME
            )

            # --- 4.2 Crossover (Operator Crossover) ---
            # 这里的 Crossover 指的是"变异算子代码之间的交叉"，逻辑与 CX 进化是一样的
            print("正在交叉生成新算子...")

            population_after_cx = await crossover(
                async_client,
                toolbox,
                population,
                hints,
                worse_codes,
                better_codes,
                directions,  # 传递方向列表
                problem_desc,
                system_prompt,
                op_type=operator_type,  # "mt"
                model_name=MODEL_NAME
            )

            # --- 4.3 评估交叉后代 ---
            print("正在评估交叉后的种群...")
            population_after_cx = evaluate_operator(population_after_cx, evaluate_pop, toolbox, env,
                                                    op_type=operator_type, seed=i)

            # --- 4.4 选择策略 (修复引用 + 语义去重 + 三分数精英更新) ---
            mixed_population = population + population_after_cx

            # a. 更新三个精英
            # 辅助函数：安全获取分数
            def get_score_safe(ind, idx):
                score = ind.get('score')
                if isinstance(score, (tuple, list)) and len(score) > idx:
                    return score[idx]
                return -float('inf')

            current_best_overall = max(mixed_population, key=lambda ind: get_score_safe(ind, 0))
            current_best_visual = max(mixed_population, key=lambda ind: get_score_safe(ind, 1))
            current_best_demand = max(mixed_population, key=lambda ind: get_score_safe(ind, 2))

            # 更新综合分精英
            if get_score_safe(current_best_overall, 0) > get_score_safe(elitist_overall, 0):
                print(f"  >>> [综合] 发现新精英！{get_score_safe(elitist_overall, 0):.4f} -> {get_score_safe(current_best_overall, 0):.4f}")
                elitist_overall = current_best_overall.copy()

            # 更新视觉分精英
            if get_score_safe(current_best_visual, 1) > get_score_safe(elitist_visual, 1):
                print(f"  >>> [视觉] 发现新精英！{get_score_safe(elitist_visual, 1):.4f} -> {get_score_safe(current_best_visual, 1):.4f}")
                elitist_visual = current_best_visual.copy()

            # 更新需求分精英
            if get_score_safe(current_best_demand, 2) > get_score_safe(elitist_demand, 2):
                print(f"  >>> [需求] 发现新精英！{get_score_safe(elitist_demand, 2):.4f} -> {get_score_safe(current_best_demand, 2):.4f}")
                elitist_demand = current_best_demand.copy()

            # b. 构建下一代 (加入去重逻辑 + 精英保护)
            next_generation = []
            seen_codes = set()

            # 1. 精英保护：保护三个精英（自动去重）
            elitists_to_protect = [elitist_overall, elitist_visual, elitist_demand]
            for elit in elitists_to_protect:
                if elit['code'] not in seen_codes:
                    next_generation.append(elit.copy())
                    seen_codes.add(elit['code'])

            # 2. 锦标赛选择填补剩余空位
            needed_count = NUM_OPERATORS - len(next_generation)
            tournament_size = 3
            attempts = 0
            max_attempts = needed_count * 5

            while len(next_generation) < NUM_OPERATORS and attempts < max_attempts:
                attempts += 1
                candidates = random.sample(mixed_population, min(len(mixed_population), tournament_size))
                # 按综合分选择优胜者
                winner = max(candidates, key=lambda ind: get_score_safe(ind, 0))

                if winner['code'] not in seen_codes:
                    ind_new = winner.copy()
                    next_generation.append(ind_new)
                    seen_codes.add(winner['code'])

            # 兜底填充
            while len(next_generation) < NUM_OPERATORS:
                candidates = random.sample(mixed_population, min(len(mixed_population), tournament_size))
                # 按综合分选择优胜者
                winner = max(candidates, key=lambda ind: get_score_safe(ind, 0))
                ind_new = winner.copy()
                next_generation.append(ind_new)

            # ================= 5. 长期反思与变异 =================

            # --- 5.1 Long-term Reflection ---
            pre_long_term_reflection = long_term_reflection if long_term_reflection != "" else ""
            long_term_reflection = gen_long_term_reflection(
                sync_client,
                pre_long_term_reflection,
                hints,
                system_prompt
            )
            print(long_term_reflection)

            # --- 5.2 Mutation (Operator Mutation) ---
            # 对三个变异算子精英分别进行代码变异
            print("正在进行变异...")

            # 对三个精英分别在三个方向上进行变异 (3 * 3 = 9 次变异)
            mutated_individuals = []

            # 定义精英来源
            elitist_sources = [
                ("OverallElit", elitist_overall),
                ("VisualElit", elitist_visual),
                ("DemandElit", elitist_demand)
            ]

            direction_prompts = {
                "Overall": "Improve Pareto Dominance and Population Diversity. Try to find a better trade-off between objectives.",
                "Visual": "Focus specifically on optimizing Objective 1 (Visual Compactness & Path Aesthetics). It is acceptable to sacrifice some Demand satisfaction.",
                "Demand": "Focus specifically on optimizing Objective 2 (Demand Satisfaction & Load Balancing). It is acceptable to sacrifice some Visual compactness."
            }

            target_directions = ["Overall", "Visual", "Demand"]

            for source_name, elit_ind in elitist_sources:
                # 安全检查：防止某个维度的精英不存在
                if not elit_ind:
                    continue

                for dir_name in target_directions:
                    # 获取详细的提示词
                    detailed_hint = direction_prompts.get(dir_name, f"Focus on {dir_name}")

                    print(f"  变异源: {source_name} -> 目标方向: {dir_name}")

                    mutated_ind = mutation(
                        sync_client,
                        elit_ind,
                        long_term_reflection,
                        base_prompt_content,
                        system_prompt,
                        env=env,
                        evaluate_pop=evaluate_pop,
                        toolbox=toolbox,
                        op_type="mt",
                        model_name=MODEL_NAME,
                        # --- 【修改点】传入详细的提示词 ---
                        direction_hint=f"[Mutation Goal: {detailed_hint}] "
                    )

                    # 修复：只匹配 mutation 失败分支的精确前缀（"# Final Error:"），
                    # 不再用 "Error" not in code 的子串匹配，避免误杀含
                    # `except ValueError` / `raise NetworkXError` 等合理代码的算子
                    if mutated_ind and not mutated_ind.get("code", "").startswith("# Final Error:"):
                        mutated_individuals.append(mutated_ind)

                        # 文件名依然使用简短的 dir_name，保持整洁
                        file_name = f"temp/mt_mutated_{source_name.lower()}_to_{dir_name.lower()}.py"
                        with open(file_name, "w", encoding="utf-8") as f:
                            f.write(mutated_ind["code"])

            # 将所有变异个体加入种群
            for mutated_ind in mutated_individuals:
                next_generation.append(mutated_ind)
            print(f"共生成 {len(mutated_individuals)} 个变异算子。")

            # ================= 6. 最终评估与收缩 =================
            print("正在进行本代最终全量评估 (触发加权平滑)...")
            evaluated_final_pop = evaluate_operator(next_generation, evaluate_pop, toolbox, env, op_type=operator_type,
                                                    seed=i)

            # 按综合分排序
            evaluated_final_pop.sort(key=lambda ind: get_score_safe(ind, 0), reverse=True)

            # 更新三个精英
            best_overall = evaluated_final_pop[0]
            best_visual = max(evaluated_final_pop, key=lambda ind: get_score_safe(ind, 1))
            best_demand = max(evaluated_final_pop, key=lambda ind: get_score_safe(ind, 2))

            if get_score_safe(best_overall, 0) > get_score_safe(elitist_overall, 0):
                print(f"  >>> [综合] 最终精英突破！{get_score_safe(elitist_overall, 0):.4f} -> {get_score_safe(best_overall, 0):.4f}")
            if get_score_safe(best_visual, 1) > get_score_safe(elitist_visual, 1):
                print(f"  >>> [视觉] 最终精英突破！{get_score_safe(elitist_visual, 1):.4f} -> {get_score_safe(best_visual, 1):.4f}")
            if get_score_safe(best_demand, 2) > get_score_safe(elitist_demand, 2):
                print(f"  >>> [需求] 最终精英突破！{get_score_safe(elitist_demand, 2):.4f} -> {get_score_safe(best_demand, 2):.4f}")

            elitist_overall = best_overall.copy()
            elitist_visual = best_visual.copy()
            elitist_demand = best_demand.copy()

            population = evaluated_final_pop[:NUM_OPERATORS]

            # 记录数据 - 计算三个维度的平均分
            valid_scores = [ind['score'] for ind in population if isinstance(ind.get('score'), (tuple, list)) and ind['score'][0] > -1000]
            if valid_scores:
                avg_overall = statistics.mean([s[0] for s in valid_scores])
                avg_visual = statistics.mean([s[1] for s in valid_scores])
                avg_demand = statistics.mean([s[2] for s in valid_scores])
            else:
                avg_overall = avg_visual = avg_demand = -1

            print(f"第 {i + 1} 代结束。")
            print(f"  精英 - 综合:{get_score_safe(elitist_overall, 0):.4f}, 视觉:{get_score_safe(elitist_visual, 1):.4f}, 需求:{get_score_safe(elitist_demand, 2):.4f}")
            print(f"  平均 - 综合:{avg_overall:.4f}, 视觉:{avg_visual:.4f}, 需求:{avg_demand:.4f}")

            history.append((elitist_overall['score'], elitist_visual['score'], elitist_demand['score'],
                            (avg_overall, avg_visual, avg_demand)))
            # 记录本代 mt 算子池精英分与平均分（统一持久化到 temp/operator_performance_iter.json）
            _operator_performance_recorder.record(
                "mt", i + 1,
                (get_score_safe(elitist_overall, 0), get_score_safe(elitist_visual, 1), get_score_safe(elitist_demand, 2)),
                (avg_overall, avg_visual, avg_demand)
            )
            try:
                with open(log_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 写入一行: [代数, 三个精英分, 三个平均分]
                    writer.writerow([i + 1,
                                     f"{get_score_safe(elitist_overall, 0):.4f}",
                                     f"{get_score_safe(elitist_visual, 1):.4f}",
                                     f"{get_score_safe(elitist_demand, 2):.4f}",
                                     f"{avg_overall:.4f}",
                                     f"{avg_visual:.4f}",
                                     f"{avg_demand:.4f}"])
            except Exception as e:
                print(f"写入日志失败: {e}")
        # ================= 7. 结束工作 =================
        print(f"\n流程结束。")
        print(f"综合精英 ID: {elitist_overall.get('idx', 'unknown')}, Score: {elitist_overall['score']}")
        print(f"视觉精英 ID: {elitist_visual.get('idx', 'unknown')}, Score: {elitist_visual['score']}")
        print(f"需求精英 ID: {elitist_demand.get('idx', 'unknown')}, Score: {elitist_demand['score']}")

        # 保存三个精英算子（T12: 双路径保存 temp/ 与 evolution_out_dir/elitists/）
        _mt_elitist_pairs = [
            ("mt_elitist_overall.py", elitist_overall),
            ("mt_elitist_visual.py", elitist_visual),
            ("mt_elitist_demand.py", elitist_demand),
        ]
        for fname, elitist_obj in _mt_elitist_pairs:
            for target_dir in ["temp", os.path.join(_mt_evolution_out_dir, "elitists")]:
                os.makedirs(target_dir, exist_ok=True)
                with open(os.path.join(target_dir, fname), "w", encoding="utf-8") as f:
                    f.write(elitist_obj['code'])

        print("精英算子已保存:")
        print(f"  - 综合: temp/mt_elitist_overall.py (Score: {elitist_overall['score']})")
        print(f"  - 视觉: temp/mt_elitist_visual.py (Score: {elitist_visual['score']})")
        print(f"  - 需求: temp/mt_elitist_demand.py (Score: {elitist_demand['score']})")

        # T12: 追踪器导出
        tracker.end_evolution()
        tracker.export_json(_mt_evolution_out_dir)
        tracker.export_csv(_mt_evolution_out_dir)
        print(f"[追踪] LLM 使用统计已保存到 {_mt_evolution_out_dir}/llm_usage.json")

        return {
            "overall": elitist_overall,
            "visual": elitist_visual,
            "demand": elitist_demand
        }  # 返回三个变异算子精英

def llm_heuristic_agent(summary_detail,target):
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus"
    # 初始化客户端
    sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    async_client = AsyncOpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)
    # 2. 定义提示词模板
    system_prompt = readText("./prompts/system_prompt.txt")
    user_prompt_template = readText("./prompts/cx_heuristic_prompt.txt") if target == "cx" else readText("./prompts/mt_heuristic_prompt.txt")

    # 3. 填充提示词并构造 Ollama 消息格式
    user_prompt = user_prompt_template.format(
        problem_desc=readText("./description/description.txt"),
        summary_detail = summary_detail,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = tracked_chat_create(sync_client, phase="other",
            model=MODEL_NAME,
            messages=messages
        )

    result = response.choices[0].message.content.strip()
    result = extract_true_response(result)
    return result

def llm_decide_agent(stage_1,stage_2,stage_3,message = None,retry = 5):
    stage_1_str = json.dumps(stage_1, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
    stage_2_str = json.dumps(stage_2, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
    stage_3_str = json.dumps(stage_3, ensure_ascii=False, indent=2)  # 转为格式化JSON字符串
    # 1. 初始化 Ollama 客户端
    # client = Client(host='http://10.108.25.202:12341')  # deepseek 服务器
    # model_name = 'deepseek-r1:14b'
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus"
    sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

    # 2. 定义提示词模板
    system_prompt = readText("./prompts/system_prompt.txt")
    user_prompt_template = readText("./prompts/decide_prompt.txt")

    # 3. 填充提示词并构造 Ollama 消息格式
    user_prompt = user_prompt_template.format(
        problem_desc = readText("./description/description.txt"),
        stage_1=stage_1_str,
        stage_2=stage_2_str,
        stage_3=stage_3_str
    )

    # prompt = prompt_template.replace("{solver}", solver.strip())
    # prompt = prompt.replace("{keyword}", keyword.strip())
    # prompt = prompt.replace("{context}", context_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if message is not None:
        messages.append({"role": "user", "content": message})
    # 4. 调用 Ollama 模型（强制 JSON 格式输出）
    response = tracked_chat_create(sync_client, phase="other",
            model=MODEL_NAME,
            messages=messages
        )

    result = response.choices[0].message.content.strip()
    print("decide返回结果：", result)
    pattern = r"\{[\s\S]*?\}"

    matches = re.findall(pattern, result)
    if matches:
        result = matches[-1]
    else:
        print("未找到字典内容")

    try:
        # 解析JSON字符串（处理可能的格式微小偏差）
        parsed = json.loads(result)

        # 校验核心字段是否存在且格式正确
        if all(key in parsed for key in ['choice', 'suggestion']) and isinstance(parsed['choice'], int) and 0 <= \
                parsed['choice'] <= 2:
            print("decide_agent返回结果：choice:", parsed['choice']," suggestion:",parsed['suggestion'].strip())
            # 确保返回格式严格匹配需求
            return parsed['choice'],parsed['suggestion'].strip()

        else:
            print("错误：返回JSON缺少核心字段或choice值无效（必须是0/1/2）")
            return {}
    except json.JSONDecodeError as e:
        print(f"错误：JSON格式解析失败 - {str(e)}")
        if retry > 0:
            print("尝试修复decide操作，修复次数：", 5 - retry)
            message = (
                "Your previous output failed JSON format parsing. The error message is: "
                f"'{str(e)}'. "
                "Please regenerate a strictly compliant JSON format output based on the evolutionary history data and system instructions you received. "
                "The content must be in the format {'choice': int, 'suggestion': str}."
            )
            return llm_decide_agent(stage_1,stage_2,stage_3,message = message,retry=retry-1)
        else:
            print("修复失败")
            return {}
    except Exception as e:
        print(f"错误：解析结果时发生异常 - {str(e)}")
        if retry > 0:
            print("尝试修复decide操作，修复次数：", 5 - retry)
            message = (
                "An exception occurred while parsing the result. The error message is: "
                f"'{str(e)}'. "
                "Please regenerate a strictly compliant JSON format output based on the evolutionary history data and system instructions you received. "
                "The content must be in the format {'choice': int, 'suggestion': str}."
            )
            return llm_decide_agent(stage_1,stage_2, stage_3,message=message,retry=retry-1)
        else:
            print("修复失败")
            return {}


def smooth_operator_weights(old_weights, new_weights, min_val=0.05, max_val=0.7, max_change=0.2):
    """
    平滑更新算子权重

    参数:
        old_weights: {"overall": 0.25, "visual": 0.25, "demand": 0.25, "default": 0.25}
        new_weights: LLM返回的原始权重
        min_val: 单个权重最小值
        max_val: 单个权重最大值
        max_change: 单次最大变化幅度

    返回:
        平滑后的权重字典
    """
    smoothed = {}
    for key in old_weights.keys():
        raw = new_weights.get(key, old_weights[key])

        # 1. 范围限制
        clipped = max(min_val, min(max_val, raw))

        # 2. 平滑过渡
        old_val = old_weights[key]
        delta = clipped - old_val
        bounded_delta = max(-max_change, min(max_change, delta))

        smoothed[key] = old_val + bounded_delta

    # 3. 归一化（确保和为1）
    total = sum(smoothed.values())
    if total > 0:
        normalized = {k: v/total for k, v in smoothed.items()}
    else:
        # 兜底：如果和为0，返回等权重
        normalized = {k: 1.0/len(old_weights) for k in old_weights.keys()}

    return normalized


def llm_select_operators(
    llm_input_info: dict,
    message: str = None,
    retry: int = 5
) -> dict:
    """
    让LLM根据双窗口对比选择算子权重

    参数:
        llm_input_info: {
            "problem_desc": str,
            "current_gen": int,
            "total_gen": int,
            "phase_status": str,
            "window1_start": int,
            "window1_end": int,
            "window1_cx_weights": str,
            "window1_mt_weights": str,
            "window1_hv_growth": float,
            "window1_hv_growth_pct": float,
            "window1_stagnation": int,
            "window1_front_change": int,
            "window1_visual_median_change": float,
            "window1_satisfy_median_change": float,
            "window1_visual_iqr_change": float,
            "window2_start": int,
            "window2_end": int,
            "window2_cx_weights": str,
            "window2_mt_weights": str,
            "window2_hv_growth": float,
            "window2_hv_growth_pct": float,
            "window2_stagnation": int,
            "window2_front_change": int,
            "window2_visual_median_change": float,
            "window2_satisfy_median_change": float,
            "window2_visual_iqr_change": float,
            "hv_growth_diff": float,
            "stagnation_diff": int,
            "next_gen_start": int,
            "next_gen_end": int
        }

    返回:
        {
            "cx_weights": {"overall": float, "visual": float, "demand": float, "default": float},
            "mt_weights": {"overall": float, "visual": float, "demand": float, "default": float},
            "phase": str,
            "rationale": str,
            "better_window": int,
            "growth_acceptable": bool
        }
    """
    ALIYUN_API_KEY = os.environ.get("ALIYUN_API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus"
    sync_client = OpenAI(api_key=ALIYUN_API_KEY, base_url=BASE_URL)

    system_prompt = readText("./prompts/system_prompt.txt")
    user_prompt_template = readText("./prompts/operator_selection_prompt.txt")

    # 填充提示词
    user_prompt = user_prompt_template.format(**llm_input_info)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if message is not None:
        messages.append({"role": "user", "content": message})

    def extract_json_block(text):
        """
        健壮的 JSON 提取函数：
        1. 去除 Markdown 代码块标记
        2. 基于括号计数查找最外层的 JSON 对象
        """
        # 1. 清理 Markdown 标记 (```json ... ```)
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()

        # 2. 寻找第一个 '{'
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # 3. 基于堆栈寻找对应的闭合 '}'
        stack = 0
        for i in range(start_idx, len(text)):
            char = text[i]
            if char == '{':
                stack += 1
            elif char == '}':
                stack -= 1
                if stack == 0:
                    # 找到最外层的闭合括号，尝试解析
                    json_str = text[start_idx: i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None
    # 调用 API
    try:
        response = tracked_chat_create(sync_client, phase="other",
            model=MODEL_NAME,
            messages=messages
        )

        result = response.choices[0].message.content.strip()
        print("\n" + "="*70)
        print("=== LLM 算子选择返回结果 ===")
        print(result)
        print("="*70 + "\n")

        # 提取并解析 JSON
        # pattern = r"\{[\s\S]*?\}"
        # matches = re.findall(pattern, result)
        parsed = extract_json_block(result)
        if parsed:

            try:

                # 验证必需字段
                required_fields = ["cx_weights", "mt_weights"]
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")

                # 验证权重格式
                for op_type in ["cx_weights", "mt_weights"]:
                    weights = parsed[op_type]
                    if not isinstance(weights, dict):
                        raise ValueError(f"{op_type} must be a dictionary")
                    if set(weights.keys()) != {"overall", "visual", "demand", "default"}:
                        raise ValueError(f"{op_type} must have keys: overall, visual, demand, default")

                print(f"[解析成功] Phase: {parsed.get('phase', 'unknown')}")
                print(f"[解析成功] Better Window: {parsed.get('better_window', 'unknown')}")
                print(f"[解析成功] Rationale: {parsed.get('rationale', 'N/A')[:100]}...")

                return parsed

            except json.JSONDecodeError as e:
                print(f"[错误] JSON解析失败: {e}")
                if retry > 0:
                    print(f"尝试修复... (剩余 {retry} 次)")
                    message = (
                        f"Your previous output failed JSON format parsing. The error message is: "
                        f"'{str(e)}'. "
                        f"Please regenerate a strictly compliant JSON format output. "
                        f"The content must be a JSON object with keys: phase, better_window, "
                        f"growth_acceptable, rationale, cx_weights, mt_weights. "
                        f"All weights must sum to 1.0."
                    )
                    return llm_select_operators(llm_input_info, message=message, retry=retry-1)
                else:
                    print("修复失败，返回空结果")
                    return {}
            except ValueError as e:
                print(f"[错误] 字段验证失败: {e}")
                if retry > 0:
                    print(f"尝试修复... (剩余 {retry} 次)")
                    message = (
                        f"Your previous output had validation errors: {str(e)}. "
                        f"Please ensure: "
                        f"1. cx_weights and mt_weights are dictionaries "
                        f"2. Each has keys: overall, visual, demand, default "
                        f"3. All weights sum to 1.0 (allow ±0.01 tolerance) "
                        f"4. Each weight is in range [0.05, 0.7]"
                    )
                    return llm_select_operators(llm_input_info, message=message, retry=retry-1)
                else:
                    print("修复失败，返回空结果")
                    return {}
        else:
            print("[错误] 未找到JSON格式的输出")
            if retry > 0:
                print(f"尝试修复... (剩余 {retry} 次)")
                message = (
                    "Your previous output did not contain a valid JSON object. "
                    "Please return ONLY a JSON object with the following structure: "
                    '{"phase": "...", "better_window": 1/2, "growth_acceptable": true/false, '
                    '"rationale": "...", "cx_weights": {...}, "mt_weights": {...}}'
                )
                return llm_select_operators(llm_input_info, message=message, retry=retry-1)
            else:
                print("修复失败，返回空结果")
                return {}

    except Exception as e:
        print(f"[错误] LLM API调用失败: {e}")
        if retry > 0:
            print(f"尝试重试... (剩余 {retry} 次)")
            message = f"The API call failed with error: {str(e)}. Please try again."
            return llm_select_operators(llm_input_info, message=message, retry=retry-1)
        else:
            print("重试失败，返回空结果")
            return {}


if __name__ == '__main__':
    from ga_engine import BusNetworkGA
    # 1. 加载数据
    edges, od, fixed_tasks, G, node_pos = load_data()

    # 2. 初始化引擎 (GA 主体)
    ga = BusNetworkGA(G, od, fixed_tasks, node_pos)

    # -----------------------------------------------------------
    # [新增步骤] 创建适配器环境 (Legacy Adapter)
    # 这个 env 对象完全符合 LLM 期望的旧版 API 文档结构
    # -----------------------------------------------------------
    print("正在构建兼容性环境 env ...")
    env = create_compatible_env(ga)

    # 3. 准备 DEAP 环境
    # 如果 BusNetworkGA 内部没有定义 Creator，需要在这里定义
    if not hasattr(creator, "MultiObjMax"):
        creator.create('MultiObjMax', base.Fitness, weights=(1.0, 1.0))
        creator.create('Individual', list, fitness=creator.MultiObjMax)

    # 4. 准备评估用的种群cond
    print("正在生成用于评估算子的临时种群...")
    # 直接使用 ga.toolbox 生成，确保个体结构正确
    # 定义保存路径
    pop_path = "evaluate_pop.pkl"

    # 如果文件存在，则直接读取
    if os.path.exists(pop_path):
        with open(pop_path, 'rb') as f:
            evaluate_pop = pickle.load(f)
        print(f"种群已从 {pop_path} 加载，共 {len(evaluate_pop)} 个个体。")
    else:
        # 如果文件不存在，则生成新的种群
        evaluate_pop = ga.toolbox.population(n=100)
        # 2. 初始评估
        invalid_ind = [ind for ind in evaluate_pop if not ind.fitness.valid]
        fitnesses = map(ga.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 保存到文件
        with open(pop_path, 'wb') as f:
            pickle.dump(evaluate_pop, f)
        print(f"种群已生成并保存到 {pop_path}。")

    # 5. 运行算子进化
    print("开始进化算子...")
    # 注意参数变化：
    # env     -> 传入适配后的 env 对象 (LLM 看得懂这个)
    # toolbox -> 传入 ga.toolbox (因为评估逻辑还在 ga 里)
    # 重置算子性能记录器：保证每次运行独立，cx/mt 两个字段均初始化为空列表
    _operator_performance_recorder.reset()
    # cx_elitist = asyncio.run(llm_gen_cx_agent(
    #     env=env,  # <--- 传入适配器对象
    #     evaluate_pop=evaluate_pop,
    #     toolbox=ga.toolbox,  # <--- 保持使用原版工具箱进行评估
    #     is_init=True
    # ))
    mt_elitist = asyncio.run(llm_gen_mt_agent(
        env=env,  # <--- 传入适配器对象
        evaluate_pop=evaluate_pop,
        toolbox=ga.toolbox,  # <--- 保持使用原版工具箱进行评估
        is_init=True
    ))
    # 统一持久化 cx/mt 分支每次迭代的精英分与平均分
    # 被注释/未运行的分支在 JSON 中以空列表占位
    _operator_performance_recorder.save("./temp/operator_performance_iter.json")
