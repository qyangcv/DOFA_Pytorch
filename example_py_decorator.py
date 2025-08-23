import time

class Timer:
    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{self.prefix}函数 {func.__name__} 执行时间: {end - start:.4f}秒")
            return result
        return wrapper

@Timer("性能测试: ")  # Timer("性能测试: ").__call__(calculate)
def calculate(n):
    return sum(range(n))

# 调用被装饰的函数
result = calculate(10000000)  # 实际调用的是 wrapper 函数