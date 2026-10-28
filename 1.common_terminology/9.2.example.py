import sys
from pathlib import Path
# 1\. Добавляем путь к папке context-compiler, чтобы Python мог импортировать наши файлы
compiler_dir = Path("/Users/shynggys.umbetov/Documents/Projects/AI/open-source/context-compiler")
if str(compiler_dir) not in sys.path:
    # sys.path.append(str(compiler_dir))
    sys.path.insert(0, str(compiler_dir))

# 2\. Импортируем наш инструмент из my_agent_tool.py
from my_agent_tool import get_compiled_code_context

# 3\. Проверяем работу инструмента вручную. 
if __name__ == "__main__": 
    print("=== ТЕСТ ИНСТРУМЕНТА AGENT TOOL ===")

    # Симулируем вызов инструмента агентом
    target = str(compiler_dir / "compiler.py")

    # Агент вызывает инструмент:
    result_prompt = get_compiled_code_context.invoke({ "target_file": target, "max_hops": 2 })
    print(f"Инструмент успешно скомпилировал контекст!")
    print(f"Первые 400 символов скомпилированного контекста:\\n")
    print(f"Result prompt: {result_prompt[:400]}")

# Запуск через команду ;
#python3 9.2.example.py

# инструмент `@tool` полностью готов к интеграции в агентов на LangChain / LangGraph

# * выведенный лог
"""
Успешный импорт ContextCompiler!
=== ТЕСТ ИНСТРУМЕНТА AGENT TOOL ===
Инструмент успешно скомпилировал контекст!
Первые 400 символов скомпилированного контекста:\n
Result prompt: # ---- [FULL SOURCE] /Users/shynggys.umbetov/Documents/Projects/AI/open-source/context-compiler/compiler.py ----

compiler.py
===========
ContextCompiler — orchestrates the three-tier context strategy.

  Tier 1  Full source            the file actively being edited
  Tier 2  Skeletonized interface  files reachable within `max_hops` call hops
  Tier 3  Total exclusion         everything else in

"""